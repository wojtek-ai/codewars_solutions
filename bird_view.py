import numpy as np
import copy

class InputPreprocessor():

  def __init__(self, mountain):
    self.mountain = mountain
    self.preprocessed_mountain = np.array([[1 if elem == '^' else 0 for elem in row] for row in self.mountain])
    self.mountain_bbox = self.make_mountain_bbox()
   
  def add_padding(self, matrix):
    return np.pad(matrix, pad_width=1, mode='constant', constant_values=0)

  def make_mountain_bbox(self, margin=True):
    mountain = self.preprocessed_mountain[~np.all(self.preprocessed_mountain == 0, axis=1)]
    mountain = np.transpose(np.transpose(mountain)[~np.all(np.transpose(mountain) == 0, axis=1)])
    if margin:
      return self.add_padding(mountain)
    else:
      return mountain


class MountainPeeker():

  def __init__(self, mountain_preprocessed):
    self.matrix = mountain_preprocessed
    self.layers_exhausted = False
    self.mountain_height = self.get_height()

  def get_non_zero_coordinates(self, matrix):
    return np.where(matrix != 0)

  def check_four_neighbours_sum(self, matrix_to_check):
    sums_dict = {}
    for coord in zip(*self.get_non_zero_coordinates(matrix_to_check)):
      x = coord[0]
      y = coord[1]
      neighbours_coords = [[x+1, y], [x-1, y], [x, y-1], [x, y+1]]
      temp_sum = 0    
      for neighbour_coord in neighbours_coords:
          temp_sum += matrix_to_check[neighbour_coord[0]][neighbour_coord[1]]
      try:
        sums_dict[temp_sum].append(coord)
      except KeyError:
        sums_dict[temp_sum] = [coord]
  
    return sums_dict 
  
  def leave_outer_mountain_only(self, matrix_to_erase):
    try:
      inner_coords = self.check_four_neighbours_sum(matrix_to_erase)[4]
    except KeyError:
      self.layers_exhausted = True
      inner_coords = None
    matrix_copy = copy.deepcopy(matrix_to_erase)
    if not inner_coords == None:
      for inner_coord in inner_coords:
        matrix_copy[inner_coord[0]][inner_coord[1]] = 0
    return matrix_copy

  def peel_single_layer(self, matrix_to_peel):
    inner_matrix = matrix_to_peel - self.leave_outer_mountain_only(matrix_to_peel)
    return inner_matrix
  
  def get_height(self):
    counter = 0 
    matrix_to_peel = self.matrix
    if np.all(matrix_to_peel == 0):
      return 0
    while not self.layers_exhausted:
      matrix_to_peel = self.peel_single_layer(matrix_to_peel)
      counter += 1
    return counter


def peak_height(mountain):
    preprocessed_mountain = InputPreprocessor(mountain).mountain_bbox
    height = MountainPeeker(preprocessed_mountain).mountain_height
    return height
