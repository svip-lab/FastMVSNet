import torch

from fastmvsnet.nn.functional import pdist


def get_pixel_grids(height, width):
    with torch.no_grad():
        # texture coordinate
        x_linspace = torch.linspace(0.5, width - 0.5, width).view(1, width).expand(height, width)
        y_linspace = torch.linspace(0.5, height - 0.5, height).view(height, 1).expand(height, width)
        # y_coordinates, x_coordinates = torch.meshgrid(y_linspace, x_linspace)
        x_coordinates = x_linspace.contiguous().view(-1)
        y_coordinates = y_linspace.contiguous().view(-1)
        ones = torch.ones(height * width)
        indices_grid = torch.stack([x_coordinates, y_coordinates, ones], dim=0)
    return indices_grid


def get_propability_map(cv, depth_map, depth_start, depth_interval):
    """get probability map from cost volume"""
    with torch.no_grad():
        batch_size, channels, height, width = list(depth_map.size())
        depth = cv.size(1)

        # byx coordinates, batched & flattened
        b_coordinates = torch.arange(batch_size, dtype=torch.int64)
        y_coordinates = torch.arange(height, dtype=torch.int64)
        x_coordinates = torch.arange(width, dtype=torch.int64)
        b_coordinates = b_coordinates.view(batch_size, 1, 1).expand(batch_size, height, width)
        y_coordinates = y_coordinates.view(1, height, 1).expand(batch_size, height, width)
        x_coordinates = x_coordinates.view(1, 1, width).expand(batch_size, height, width)

        b_coordinates = b_coordinates.contiguous().view(-1).type(torch.long)
        y_coordinates = y_coordinates.contiguous().view(-1).type(torch.long)
        x_coordinates = x_coordinates.contiguous().view(-1).type(torch.long)
        # b_coordinates = _repeat_(b_coordinates, batch_size)
        # y_coordinates = _repeat_(y_coordinates, batch_size)
        # x_coordinates = _repeat_(x_coordinates, batch_size)

        # d coordinates (floored and ceiled), batched & flattened
        d_coordinates = ((depth_map - depth_start.view(-1, 1, 1, 1)) / depth_interval.view(-1, 1, 1, 1)).view(-1)
        d_coordinates = torch.detach(d_coordinates)
        d_coordinates_left0 = torch.clamp(d_coordinates.floor(), 0, depth - 1).type(torch.long)
        d_coordinates_right0 = torch.clamp(d_coordinates.ceil(), 0, depth - 1).type(torch.long)

        # # get probability image by gathering
        prob_map_left0 = cv[b_coordinates, d_coordinates_left0, y_coordinates, x_coordinates]
        prob_map_right0 = cv[b_coordinates, d_coordinates_right0, y_coordinates, x_coordinates]

        prob_map = prob_map_left0 + prob_map_right0
        prob_map = prob_map.view(batch_size, 1, height, width)

    return prob_map
