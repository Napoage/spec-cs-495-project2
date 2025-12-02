# Mock hardware components for local testing
class MockMPU9250:
    def __init__(self, *args, **kwargs):
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0

    def configure(self):
        pass

    def begin(self):
        pass

    def loadCalibDataFromFile(self, filepath):
        pass

    def readSensor(self):
        pass

    def computeOrientation(self):
        pass

    def readGyroscope(self):
        return [0.0, 0.0, 0.0]

    def readAccelerometer(self):
        return [0.0, 0.0, 1.0]

    def readMagnetometer(self):
        return [0.0, 0.0, 0.0]

# Mock smbus
class MockSMBus:
    def __init__(self, *args):
        pass

    def read_byte_data(self, *args):
        return 0

    def write_byte_data(self, *args):
        pass

# Replace hardware modules
import sys
sys.modules['smbus'] = type('MockModule', (), {'SMBus': MockSMBus})()
sys.modules['imusensor'] = type('MockModule', (), {})()
sys.modules['imusensor.MPU9250'] = type('MockModule', (), {'MPU9250': MockMPU9250})()