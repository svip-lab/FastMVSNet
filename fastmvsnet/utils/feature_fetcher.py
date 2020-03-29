import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


class FeatureFetcher(nn.Module):
    def __init__(self, mode="bilinear"):
        super(FeatureFetcher, self).__init__()
        self.mode = mode

    def forward(self, feature_maps, pts, cam_intrinsics, cam_extrinsics):
        """

        :param feature_maps: torch.tensor, [B, V, C, H, W]
        :param pts: torch.tensor, [B, 3, N]
        :param cam_intrinsics: torch.tensor, [B, V, 3, 3]
        :param cam_extrinsics: torch.tensor, [B, V, 3, 4], [R|t], p_cam = R*p_world + t
        :return:
            pts_feature: torch.tensor, [B, V, C, N]
        """
        batch_size, num_view, channels, height, width = list(feature_maps.size())
        feature_maps = feature_maps.view(batch_size * num_view, channels, height, width)

        curr_batch_size = batch_size * num_view
        cam_intrinsics = cam_intrinsics.view(curr_batch_size, 3, 3)

        with torch.no_grad():
            num_pts = pts.size(2)
            pts_expand = pts.unsqueeze(1).contiguous().expand(batch_size, num_view, 3, num_pts) \
                .contiguous().view(curr_batch_size, 3, num_pts)
            if cam_extrinsics is None:
                transformed_pts = pts_expand.type(torch.float).transpose(1, 2)
            else:
                cam_extrinsics = cam_extrinsics.view(curr_batch_size, 3, 4)
                R = torch.narrow(cam_extrinsics, 2, 0, 3)
                t = torch.narrow(cam_extrinsics, 2, 3, 1).expand(curr_batch_size, 3, num_pts)
                transformed_pts = torch.bmm(R, pts_expand) + t
                transformed_pts = transformed_pts.type(torch.float).transpose(1, 2)
            x = transformed_pts[..., 0]
            y = transformed_pts[..., 1]
            z = transformed_pts[..., 2]

            normal_uv = torch.cat(
                [torch.div(x, z).unsqueeze(-1), torch.div(y, z).unsqueeze(-1), torch.ones_like(x).unsqueeze(-1)],
                dim=-1)
            uv = torch.bmm(normal_uv, cam_intrinsics.transpose(1, 2))
            uv = uv[:, :, :2]

            grid = (uv - 0.5).view(curr_batch_size, num_pts, 1, 2)
            grid[..., 0] = (grid[..., 0] / float(width - 1)) * 2 - 1.0
            grid[..., 1] = (grid[..., 1] / float(height - 1)) * 2 - 1.0

        # pts_feature = F.grid_sample(feature_maps, grid, mode=self.mode, padding_mode='border')
        # print("without border pad-----------------------")
        pts_feature = F.grid_sample(feature_maps, grid, mode=self.mode)
        pts_feature = pts_feature.squeeze(3)

        pts_feature = pts_feature.view(batch_size, num_view, channels, num_pts)

        return pts_feature


class FeatureGradFetcher(nn.Module):
    def __init__(self, mode="bilinear"):
        super(FeatureGradFetcher, self).__init__()
        self.mode = mode

    def forward(self, feature_maps, pts, cam_intrinsics, cam_extrinsics):
        """

        :param feature_maps: torch.tensor, [B, V, C, H, W]
        :param pts: torch.tensor, [B, 3, N]
        :param cam_intrinsics: torch.tensor, [B, V, 3, 3]
        :param cam_extrinsics: torch.tensor, [B, V, 3, 4], [R|t], p_cam = R*p_world + t
        :return:
            pts_feature: torch.tensor, [B, V, C, N]
        """
        batch_size, num_view, channels, height, width = list(feature_maps.size())
        feature_maps = feature_maps.view(batch_size * num_view, channels, height, width)

        curr_batch_size = batch_size * num_view
        cam_intrinsics = cam_intrinsics.view(curr_batch_size, 3, 3)

        with torch.no_grad():
            num_pts = pts.size(2)
            pts_expand = pts.unsqueeze(1).contiguous().expand(batch_size, num_view, 3, num_pts) \
                .contiguous().view(curr_batch_size, 3, num_pts)
            if cam_extrinsics is None:
                transformed_pts = pts_expand.type(torch.float).transpose(1, 2)
            else:
                cam_extrinsics = cam_extrinsics.view(curr_batch_size, 3, 4)
                R = torch.narrow(cam_extrinsics, 2, 0, 3)
                t = torch.narrow(cam_extrinsics, 2, 3, 1).expand(curr_batch_size, 3, num_pts)
                transformed_pts = torch.bmm(R, pts_expand) + t
                transformed_pts = transformed_pts.type(torch.float).transpose(1, 2)
            x = transformed_pts[..., 0]
            y = transformed_pts[..., 1]
            z = transformed_pts[..., 2]

            normal_uv = torch.cat(
                [torch.div(x, z).unsqueeze(-1), torch.div(y, z).unsqueeze(-1), torch.ones_like(x).unsqueeze(-1)],
                dim=-1)
            uv = torch.bmm(normal_uv, cam_intrinsics.transpose(1, 2))
            uv = uv[:, :, :2]

            grid = (uv - 0.5).view(curr_batch_size, num_pts, 1, 2)
            grid[..., 0] = (grid[..., 0] / float(width - 1)) * 2 - 1.0
            grid[..., 1] = (grid[..., 1] / float(height - 1)) * 2 - 1.0

            #todo check bug
            grid_l = grid.clone()
            grid_l[..., 0] -= (1. / float(width - 1)) * 2

            grid_r = grid.clone()
            grid_r[..., 0] += (1. / float(width - 1)) * 2

            grid_t = grid.clone()
            grid_t[..., 1] -= (1. / float(height - 1)) * 2

            grid_b = grid.clone()
            grid_b[..., 1] += (1. / float(height - 1)) * 2


        def get_features(grid_uv):
            pts_feature = F.grid_sample(feature_maps, grid_uv, mode=self.mode)
            pts_feature = pts_feature.squeeze(3)

            pts_feature = pts_feature.view(batch_size, num_view, channels, num_pts)
            return pts_feature

        pts_feature = get_features(grid)

        pts_feature_l = get_features(grid_l)
        pts_feature_r = get_features(grid_r)
        pts_feature_t = get_features(grid_t)
        pts_feature_b = get_features(grid_b)

        pts_feature_grad_x = 0.5 * (pts_feature_r - pts_feature_l)
        pts_feature_grad_y = 0.5 * (pts_feature_b - pts_feature_t)

        pts_feature_grad = torch.stack((pts_feature_grad_x, pts_feature_grad_y), dim=-1)
        # print("================features++++++++++++")
        # print(feature_maps)
        # print ("===========grad+++++++++++++++")
        # print (pts_feature_grad)
        return pts_feature, pts_feature_grad

    def get_result(self,  feature_maps, pts, cam_intrinsics, cam_extrinsics):
        batch_size, num_view, channels, height, width = list(feature_maps.size())
        feature_maps = feature_maps.view(batch_size * num_view, channels, height, width)

        curr_batch_size = batch_size * num_view
        cam_intrinsics = cam_intrinsics.view(curr_batch_size, 3, 3)

        num_pts = pts.size(2)
        pts_expand = pts.unsqueeze(1).contiguous().expand(batch_size, num_view, 3, num_pts) \
            .contiguous().view(curr_batch_size, 3, num_pts)
        if cam_extrinsics is None:
            transformed_pts = pts_expand.type(torch.float).transpose(1, 2)
        else:
            cam_extrinsics = cam_extrinsics.view(curr_batch_size, 3, 4)
            R = torch.narrow(cam_extrinsics, 2, 0, 3)
            t = torch.narrow(cam_extrinsics, 2, 3, 1).expand(curr_batch_size, 3, num_pts)
            transformed_pts = torch.bmm(R, pts_expand) + t
            transformed_pts = transformed_pts.type(torch.float).transpose(1, 2)
        x = transformed_pts[..., 0]
        y = transformed_pts[..., 1]
        z = transformed_pts[..., 2]

        normal_uv = torch.cat(
            [torch.div(x, z).unsqueeze(-1), torch.div(y, z).unsqueeze(-1), torch.ones_like(x).unsqueeze(-1)],
            dim=-1)
        uv = torch.bmm(normal_uv, cam_intrinsics.transpose(1, 2))
        uv = uv[:, :, :2]

        grid = (uv - 0.5).view(curr_batch_size, num_pts, 1, 2)
        grid[..., 0] = (grid[..., 0] / float(width - 1)) * 2 - 1.0
        grid[..., 1] = (grid[..., 1] / float(height - 1)) * 2 - 1.0

        def get_features(grid_uv):
            pts_feature = F.grid_sample(feature_maps, grid_uv, mode=self.mode)
            pts_feature = pts_feature.squeeze(3)

            pts_feature = pts_feature.view(batch_size, num_view, channels, num_pts)
            return pts_feature.detach()

        pts_feature = get_features(grid)

        # todo check bug
        grid[..., 0] -= (1. / float(width - 1)) * 2
        pts_feature_l = get_features(grid)
        grid[..., 0] += (1. / float(width - 1)) * 2

        grid[..., 0] += (1. / float(width - 1)) * 2
        pts_feature_r = get_features(grid)
        grid[..., 0] -= (1. / float(width - 1)) * 2

        grid[..., 1] -= (1. / float(height - 1)) * 2
        pts_feature_t = get_features(grid)
        grid[..., 1] += (1. / float(height - 1)) * 2

        grid[..., 1] += (1. / float(height - 1)) * 2
        pts_feature_b = get_features(grid)
        grid[..., 1] -= (1. / float(height - 1)) * 2

        pts_feature_r -= pts_feature_l
        pts_feature_r *= 0.5
        pts_feature_b -= pts_feature_t
        pts_feature_b *= 0.5

        return pts_feature.detach(), pts_feature_r.detach(), pts_feature_b.detach()

    def test_forward(self, feature_maps, pts, cam_intrinsics, cam_extrinsics):
        """

        :param feature_maps: torch.tensor, [B, V, C, H, W]
        :param pts: torch.tensor, [B, 3, N]
        :param cam_intrinsics: torch.tensor, [B, V, 3, 3]
        :param cam_extrinsics: torch.tensor, [B, V, 3, 4], [R|t], p_cam = R*p_world + t
        :return:
            pts_feature: torch.tensor, [B, V, C, N]
        """
        with torch.no_grad():
            pts_feature, grad_x, grad_y = \
                self.get_result(feature_maps, pts, cam_intrinsics, cam_extrinsics)
        torch.cuda.empty_cache()
        pts_feature_grad = torch.stack((grad_x, grad_y), dim=-1)

        return pts_feature.detach(), pts_feature_grad.detach()


class PointGrad(nn.Module):
    def __init__(self):
        super(PointGrad, self).__init__()

    def forward(self, pts, cam_intrinsics, cam_extrinsics):
        """
        :param pts: torch.tensor, [B, 3, N]
        :param cam_intrinsics: torch.tensor, [B, V, 3, 3]
        :param cam_extrinsics: torch.tensor, [B, V, 3, 4], [R|t], p_cam = R*p_world + t
        :return:
            pts_feature: torch.tensor, [B, V, C, N]
        """
        batch_size, num_view, _, _ = list(cam_extrinsics.size())

        curr_batch_size = batch_size * num_view
        cam_intrinsics = cam_intrinsics.view(curr_batch_size, 3, 3)

        with torch.no_grad():
            num_pts = pts.size(2)
            pts_expand = pts.unsqueeze(1).contiguous().expand(batch_size, num_view, 3, num_pts) \
                .contiguous().view(curr_batch_size, 3, num_pts)
            if cam_extrinsics is None:
                transformed_pts = pts_expand.type(torch.float).transpose(1, 2)
            else:
                cam_extrinsics = cam_extrinsics.view(curr_batch_size, 3, 4)
                R = torch.narrow(cam_extrinsics, 2, 0, 3)
                t = torch.narrow(cam_extrinsics, 2, 3, 1).expand(curr_batch_size, 3, num_pts)
                transformed_pts = torch.bmm(R, pts_expand) + t
                transformed_pts = transformed_pts.type(torch.float).transpose(1, 2)
            x = transformed_pts[..., 0]
            y = transformed_pts[..., 1]
            z = transformed_pts[..., 2]

            fx = cam_intrinsics[..., 0, 0].view(curr_batch_size, 1)
            fy = cam_intrinsics[..., 1, 1].view(curr_batch_size, 1)

            # print("x", x.size())
            # print("fx", fx.size(), fx, fy)

            zero = torch.zeros_like(x)
            grad_u = torch.stack([fx / z, zero, -fx * x / (z**2)], dim=-1)
            grad_v = torch.stack([zero, fy / z, -fy * y / (z**2)], dim=-1)
            grad_p = torch.stack((grad_u, grad_v), dim=-2)
            # print("grad_u size:", grad_u.size())
            # print("grad_p size:", grad_p.size())
            grad_p = grad_p.view(batch_size, num_view, num_pts, 2, 3)
        return grad_p



class ProjectUVFetcher(nn.Module):
    def __init__(self, mode="bilinear"):
        super(ProjectUVFetcher, self).__init__()
        self.mode = mode

    def forward(self, pts, cam_intrinsics, cam_extrinsics):
        """

        :param pts: torch.tensor, [B, 3, N]
        :param cam_intrinsics: torch.tensor, [B, V, 3, 3]
        :param cam_extrinsics: torch.tensor, [B, V, 3, 4], [R|t], p_cam = R*p_world + t
        :return:
            pts_feature: torch.tensor, [B, V, C, N]
        """
        batch_size, num_view = cam_extrinsics.size()[:2]

        curr_batch_size = batch_size * num_view
        cam_intrinsics = cam_intrinsics.view(curr_batch_size, 3, 3)

        with torch.no_grad():
            num_pts = pts.size(2)
            pts_expand = pts.unsqueeze(1).contiguous().expand(batch_size, num_view, 3, num_pts) \
                .contiguous().view(curr_batch_size, 3, num_pts)
            if cam_extrinsics is None:
                transformed_pts = pts_expand.type(torch.float).transpose(1, 2)
            else:
                cam_extrinsics = cam_extrinsics.view(curr_batch_size, 3, 4)
                R = torch.narrow(cam_extrinsics, 2, 0, 3)
                t = torch.narrow(cam_extrinsics, 2, 3, 1).expand(curr_batch_size, 3, num_pts)
                transformed_pts = torch.bmm(R, pts_expand) + t
                transformed_pts = transformed_pts.type(torch.float).transpose(1, 2)
            x = transformed_pts[..., 0]
            y = transformed_pts[..., 1]
            z = transformed_pts[..., 2]

            normal_uv = torch.cat(
                [torch.div(x, z).unsqueeze(-1), torch.div(y, z).unsqueeze(-1), torch.ones_like(x).unsqueeze(-1)],
                dim=-1)
            uv = torch.bmm(normal_uv, cam_intrinsics.transpose(1, 2))
            uv = uv[:, :, :2]

            grid = (uv - 0.5).view(curr_batch_size, num_pts, 1, 2)

        return grid.view(batch_size, num_view, num_pts, 1, 2)


def test_feature_fetching():
    import numpy as np
    batch_size = 3
    num_view = 2
    channels = 16
    height = 240
    width = 320
    num_pts = 32

    cam_intrinsic = torch.tensor([[10, 0, 1], [0, 10, 1], [0, 0, 1]]).float() \
        .view(1, 1, 3, 3).expand(batch_size, num_view, 3, 3).cuda()
    cam_extrinsic = torch.rand(batch_size, num_view, 3, 4).cuda()

    feature_fetcher = FeatureFetcher().cuda()

    features = torch.rand(batch_size, num_view, channels, height, width).cuda()

    imgpt = torch.tensor([60.5, 80.5, 1.0]).view(1, 1, 3, 1).expand(batch_size, num_view, 3, num_pts).cuda()

    z = 200

    pt = torch.matmul(torch.inverse(cam_intrinsic), imgpt) * z

    pt = torch.matmul(torch.inverse(cam_extrinsic[:, :, :, :3]),
                      (pt - cam_extrinsic[:, :, :, 3].unsqueeze(-1)))  # Xc = [R|T] Xw

    gathered_feature = feature_fetcher(features, pt[:, 0, :, :], cam_intrinsic, cam_extrinsic)

    gathered_feature = gathered_feature[:, 0, :, 0]
    np.savetxt("gathered_feature.txt", gathered_feature.detach().cpu().numpy(), fmt="%.4f")

    groundtruth_feature = features[:, :, :, 80, 60][:, 0, :]
    np.savetxt("groundtruth_feature.txt", groundtruth_feature.detach().cpu().numpy(), fmt="%.4f")

    print(np.allclose(gathered_feature.detach().cpu().numpy(), groundtruth_feature.detach().cpu().numpy(), 1.e-2))


if __name__ == "__main__":
    test_feature_fetching()
