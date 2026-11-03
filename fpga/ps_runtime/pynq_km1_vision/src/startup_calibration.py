"""Reuse validation of one immutable camera/station; poses stay per-frame."""
import copy


class CalibrationValues:
    def __init__(self, camera, station, camera_parameters, station_parameters):
        self.camera, self.station = copy.deepcopy(camera), copy.deepcopy(station)
        self.original_camera, self.original_station = camera_parameters, station_parameters
        self.camera_values = camera_parameters(camera)
        self.station_values = station_parameters(station)

    def camera_parameters(self, camera):
        if type(camera) is dict and camera == self.camera:
            return copy.deepcopy(self.camera_values)
        return self.original_camera(camera)

    def station_parameters(self, station):
        if type(station) is dict and station == self.station:
            return copy.deepcopy(self.station_values)
        return self.original_station(station)
