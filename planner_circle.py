import math
import matplotlib.pyplot as plt
import numpy as np
from queue import PriorityQueue

class Line:
    # line defined by p1, p2; each point = (X, Y)
    # boundType:
    # ** information of what type of boundary is it? See valid inputs below
    # "T", "B", "R", "L" - only a Top, Bottom, Right, or Left line respectively
    # "TL" or "LT" - Top and Left line
    # "TR" or "RT" - Top and Right line
    # "BL" or "LB" - Bottom and Left line
    # "BR" or "RB" - Bottom and Right line
    def __init__(self, p1, p2, boundType):
        self.p1 = p1
        self.p2 = p2
        self.boundType = boundType
        pass

    # This function, given an (x, y) coordinate,
    # the line checks to see if, relative to itself, the said coordinate is
    # Above itself (in the case of it having boundType "B", ie bottom line)
    # Below itself (in the case of it having boundType "T", ie Top line)
    # Right of itself (in the case of it having boundType "L", ie Left line)
    # or Left of itself (in the case of it having boundType "R", ie Right line)
    def inline(self, x, y):
        inLine = True
        y_on_line = 0  # gets changed
        x_on_line = 0  # gets changed

        # in the case of a vertical line, correct code shouldn't input "T" or "B"
        # so we don't care about "y_on_line"
        if self.p1[0] == self.p2[0]:
            y_on_line = 0
            x_on_line = self.p1[0]
        # in the case of a horzontal line, correct code shouldn't input "R" or "L"
        # so we don't care about "x_on_line"
        elif self.p1[1] == self.p2[1]:
            y_on_line = self.p1[1]
            x_on_line = 0
        # if not horizontal or vertical, then we do math to get relative xs and ys
        else:
            y_on_line = self.p1[1] + (self.p2[1] - self.p1[1]) * (x - self.p1[0]) / (self.p2[0] - self.p1[0])
            x_on_line = self.p1[0] + (self.p2[0] - self.p1[0]) * (y - self.p1[1]) / (self.p2[1] - self.p1[1])

        # using x_on_line and y_on_line, we can now easily check if the point is "within" the line
        for char in self.boundType:
            if char == 'T':
                inLine = inLine and (y < y_on_line)
            if char == 'B':
                inLine = inLine and (y > y_on_line)
            if char == 'R':
                inLine = inLine and (x < x_on_line)
            if char == 'L':
                inLine = inLine and (x > x_on_line)
            # print(char)
            # print(inLine)

        return inLine


class Obstacle:
    # all objects are of object type "Line"
    def __init__(self, l1, l2, l3, l4):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        pass

    # returns true if x,y is in the obstacle, false otherwise
    def clash(self, x, y):
        inBounds = True
        inBounds = inBounds and self.l1.inline(x, y)
        inBounds = inBounds and self.l2.inline(x, y)
        inBounds = inBounds and self.l3.inline(x, y)
        inBounds = inBounds and self.l4.inline(x, y)

        return inBounds


def construct_obstacles(isEasy):
    # construct obstacles
    # Remeber that points are (X, Y)

    easyObstacles = [
        Obstacle(
            Line((8, 12), (14, 12), "B"),
            Line((14, 12), (14, 18), "R"),
            Line((14, 18), (8, 18), "T"),
            Line((8, 18), (8, 12), "L")
        ),
        Obstacle(
            Line((18, 12), (18 + 6, 12), "B"),
            Line((18 + 6, 12), (18 + 6, 12 + 18), "R"),
            Line((18 + 6, 12 + 18), (18, 12 + 18), "T"),
            Line((18, 12 + 18), (18, 12), "L")
        ),
        Obstacle(
            Line((16, 44), (16 + 6, 44), "B"),
            Line((16 + 6, 44), (16 + 6, 44 + 6), "R"),
            Line((16 + 6, 44 + 6), (16, 44 + 6), "T"),
            Line((16, 44 + 6), (16, 44), "L")
        ),
        Obstacle(
            Line((38, 36), (38 + 18, 36), "B"),
            Line((38 + 18, 36), (38 + 18, 36 + 6), "R"),
            Line((38 + 18, 36 + 6), (38, 36 + 6), "T"),
            Line((38, 36 + 6), (38, 36), "L")
        )
    ]
    hardObstacles = [
        Obstacle(
            Line((44, 4), (44 + 18 / math.sqrt(2), 4 + 18 / math.sqrt(2)), "BR"),
            Line((44 + 18 / math.sqrt(2), 4 + 18 / math.sqrt(2)),
                 (44 + 18 / math.sqrt(2) - 6 / math.sqrt(2), 4 + 18 / math.sqrt(2) + 6 / math.sqrt(2)), "TR"),
            Line((44, 4), (44 - 6 / math.sqrt(2), 4 + 6 / math.sqrt(2)), "LB"),
            Line((44 - 6 / math.sqrt(2), 4 + 6 / math.sqrt(2)),
                 (44 - 6 / math.sqrt(2) + 18 / math.sqrt(2), 4 + 6 / math.sqrt(2) + 18 / math.sqrt(2)), "LT")
        ),
        Obstacle(
            Line((23, 19), (23 + 10.25, 19), "B"),
            Line((23 + 10.25, 19), (23 + 10.25, 19 + 9), "R"),
            Line((23 + 10.25, 19 + 9), (23, 19 + 9), "T"),
            Line((23, 19 + 9), (23, 19), "L")
        ),
        Obstacle(
            Line((6, 34.5), (6 + 6, 34.5), "B"),
            Line((6 + 6, 34.5), (6 + 6, 34.5 + 6), "R"),
            Line((6 + 6, 34.5 + 6), (6, 34.5 + 6), "T"),
            Line((6, 34.5 + 6), (6, 34.5), "L")
        ),
        Obstacle(
            Line((40, 50), (40 + 18 / math.sqrt(2), 50 - 18 / math.sqrt(2)), "TR"),
            Line((40 + 18 / math.sqrt(2), 50 - 18 / math.sqrt(2)),
                 (40 + 18 / math.sqrt(2) - 6 / math.sqrt(2), 50 - 18 / math.sqrt(2) - 6 / math.sqrt(2)), "RB"),
            Line((40, 50), (40 - 6 / math.sqrt(2), 50 - 6 / math.sqrt(2)), "LT"),
            Line((40 - 6 / math.sqrt(2), 50 - 6 / math.sqrt(2)),
                 (40 - 6 / math.sqrt(2) + 18 / math.sqrt(2), 50 - 6 / math.sqrt(2) - 18 / math.sqrt(2)), "LB")
        )
    ]

    if isEasy:
        obstacles = easyObstacles
    else:
        obstacles = hardObstacles  # or hardObstacles
    return obstacles  # array of obstacles


# given a list of obstacles and a coordinate point,
# check if it is within the bounds of any obstacle
def check_obstacles(obstacles, x, y):
    for ob in obstacles:
        if ob.clash(x, y):
            return True
    return False

"""
Constructs a C-Space of the map. Approximates the robot to the circle that circumscribes the robot
"""
def construct_map(isEasy, resolution, bot_radius):
    obstacles = construct_obstacles(isEasy)

    # Discritize points and run through them:

    # @TODO Vary RESOLUTION as desired if
    # The larger it is, the more accurate your map will be,
    # but the longer it will take

    # RESOLUTION is the number of points you want per inch
    # (anything larger than 20 takes a while)
    RESOLUTION = resolution
    # WIDTH and HEIGHT in inches of the course
    MAP_WIDTH = 72
    MAP_HEIGHT = 54

    x_disc = MAP_WIDTH * RESOLUTION
    y_disc = MAP_HEIGHT * RESOLUTION

    bot_radius *= RESOLUTION

    # Discritized Image
    # numpy does y first, then x

    img = np.zeros((y_disc, x_disc), dtype=np.uint8)

    for col in range(x_disc):
        for row in range(y_disc):
            hit = check_obstacles(obstacles, col / RESOLUTION, row / RESOLUTION)
            img[row, col] = hit

    cspace = img.copy()

    for col in range(x_disc):
        for row in range(y_disc):
            y, x = np.ogrid[:y_disc, :x_disc]
            dist_from_center = np.sqrt((x - col) ** 2 + (y - row) ** 2)
            mask = dist_from_center <= bot_radius

            masked = img.copy()
            masked[~mask] = 0

            if np.any(masked):
                cspace[row, col] = 1

    for col in range(x_disc):
        for row in range(y_disc):
            if (row + bot_radius > y_disc or col + bot_radius > x_disc
                    or row - bot_radius < 0 or col - bot_radius < 0):
                cspace[row, col] = 1

    return cspace

def construct_path(img, start, end):
    DIRECTIONS = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, 1, math.sqrt(2))]
    turn_penalty = 0.6

    if start == end:
        return []

    if img[start[0], start[1]] == 1 or img[end[0], end[1]] == 1:
        raise Exception("Path cannot exist! Endpoints invalid!")

    height, width = img.shape

    pq = PriorityQueue()
    pq.put((0, start, None)) # Priority queue of cost, node, and last direction

    distances = {(start, None): 0}  # Store distances with explicit weight tracking
    bp = {end: None}

    while not pq.empty():
        distance, node, last_direction = pq.get()
        r, c = node

        if node == end:
            break

        for dr, dc, move_cost in DIRECTIONS:
            nr, nc = r + dr, c + dc
            new_direction = (dr, dc)
            new_node = (nr, nc)

            if 0 <= nr < height and 0 <= nc < width and img[nr, nc] == 0:
                if last_direction is not None and new_direction != last_direction:
                    move_cost += turn_penalty  # Apply turn penalty when changing direction

                new_distance = distance + move_cost

                if (new_node, new_direction) not in distances or new_distance < distances[(new_node, new_direction)]:
                    distances[(new_node, new_direction)] = new_distance
                    bp[new_node] = node  # Store path
                    pq.put((new_distance, new_node, new_direction))

    print("Finished path!")
    # Reconstruct path
    path = []
    current = end
    while current in bp:
        path.append(current)
        current = bp[current]
        print(path)

    # Add start to path and reverse to get correct order
    path.append(start)
    path.reverse()

    return path if path[0] == start else None

def cleanup_path(path):
    compressed_path = []

    path_len = len(path)

    i = 0
    prev_dx, prev_dy = 0, 0

    for i in range(path_len - 1):
        curr = path[i]
        next_elem = path[i + 1]

        dx, dy = next_elem[0] - curr[0], next_elem[1] - curr[1]

        if (prev_dx,prev_dy) != (dx, dy):
            compressed_path.append(curr)
            prev_dx, prev_dy = dx, dy

    return compressed_path


def main():
    bot_radius = 4 #TODO: Change to match actual bot radius

    start = input("Enter starting set of coordinates, seperated by a space: ")
    end = input("Enter ending set of coordinates, seperated by a space: ")

    start_x, start_y = tuple(map(int, start.split(" ")))
    end_x, end_y = tuple(map(int, end.split(" ")))

    # Example plt code using img result from construct_map:
    isEasy = False

    # Don't have less than 1 resolution...
    # RESOLUTION is the number of points you want per inch
    # (anything larger than 20 takes a while)
    resolution = 4
    img = construct_map(isEasy, resolution, bot_radius)
    y_idx, x_idx = np.where(img == 1)

    # plt.imshow(img, cmap=plt.get_cmap('gray'), origin='lower')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Scatter plot of occupied configurations
    ax.scatter(x_idx, y_idx, c='black', marker='o', alpha=0.5)

    # Labels and aesthetics
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    path = construct_path(img, (start_y * resolution, start_x * resolution), (end_y * resolution, end_x * resolution))

    if path is None:
        print("No path found!")
        return

    path_y = [y for x, y in path]
    path_x = [x for x, y in path]

    # print(path)

    ax.scatter(path_y, path_x, c='blue', marker='o', alpha=0.5)
    plt.show()

    compressed_path = cleanup_path(path)
    print(compressed_path)

if __name__ == '__main__':
    main()