from bluesky.simulators import est_time
from ophyd.sim import hw
from bluesky.run_engine import RunEngine
from ophyd.telemetry import TelemetryUI
from bluesky.plans import scan, count
from bluesky.plan_stubs import mv, sleep
import pytest
import numpy.testing

hw = hw()



###Start by testing that the telemetry recording is working

#test telemetry recording.
def test_telemetry_recording(RE, hw):
    motor1 = hw.motor1
    det1 = hw.det1
    
    length = len(TelemetryUI.telemetry) #find the current length of the telemetry dictionary
    RE(mv(motor1, 5)) #move motor1
    assert len(TelemetryUI.telemetry) == length + 1 #check an entry has been added to the dictionary

    length = len(TelemetryUI.telemetry) #find the current length of the telemetry dictionary
    RE(count([det1])) #stage, trigger and unstage det 1
    assert len(TelemetryUI.telemetry) == length + 1 #check an entry has been added to the dictionary 
                                                   #(3 once stage and unstage are added)


#set up the telemetry dictionary with some known values.
@pytest.fixture 
def setup_telemetry():
    TelemetryUI.telemetry = [] #clear out anything in the telemetry dictionary.
    TelemetryUI.telemetry.extend([{'action': 'set', 'estimation': {'std_dev': float('nan'), 'time': 3.0}, 
                                  'object_name': 'motor1', 'position': {'start': 0, 'stop': 3}, 
                                  'settle_time': {'setpoint': 0}, 'time': {'start': 10, 'stop': 12.85}, 
                                  'velocity': {'setpoint': 1}},
                                 {'action': 'set', 'estimation': {'std_dev': float('nan'), 'time': 0.95}, 
                                  'object_name': 'motor1', 'position': {'start': 3, 'stop': 2}, 
                                  'settle_time': {'setpoint': 0}, 'time': {'start': 15, 'stop': 16.05}, 
                                  'velocity': {'setpoint': 1}}]) 
                                                            #add 2 'set' values for motor 1 to telemetry.

    TelemetryUI.telemetry.extend([{'action': 'stage', 
                                  'estimation':{'std_dev': float('nan'), 'time': float('nan')},
                                  'object_name': 'motor1', 'time': {'start': 20, 'stop': 21.05}},
                                 {'action': 'stage', 
                                  'estimation':{'std_dev': float('nan'), 'time': float('nan')},
                                  'object_name': 'motor1', 'time': {'start': 22, 'stop': 22.95}},
                                 {'action': 'unstage', 
                                  'estimation':{'std_dev': float('nan'), 'time': float('nan')},
                                  'object_name': 'motor1', 'time': {'start': 20, 'stop': 21.05}},
                                 {'action': 'unstage', 
                                  'estimation':{'std_dev': float('nan'), 'time': float('nan')},
                                  'object_name': 'motor1', 'time': {'start': 22, 'stop': 22.95}}])
                                                                #add 2 'stage/unstage' values for motor1

    TelemetryUI.telemetry.extend([{'acquire_period': {'setpoint': 1}, 'acquire_time': {'setpoint': 1},
                                  'action': 'trigger', 
                                  'estimation': {'std_dev': float('nan'), 'time': 1.0},
                                  'num_images': {'setpoint': 1}, 'object_name': 'det1',
                                  'settle_time': {'setpoint': 0}, 'time': {'start': 20, 'stop': 21.05},
                                  'trigger_mode': {'setpoint': 1}},
                                 {'acquire_period': {'setpoint': 1}, 'acquire_time': {'setpoint': 1.0},
                                  'action': 'trigger', 
                                  'estimation': {'std_dev': float('nan'), 'time': 1.05},
                                  'num_images': {'setpoint': 1}, 'object_name': 'det1',
                                  'settle_time': {'setpoint': 0}, 'time': {'start': 22, 'stop': 22.95},
                                  'trigger_mode': {'setpoint': 1}}]) #add 2 'trigger' values for det1

    TelemetryUI.telemetry.extend([{'action': 'stage', 
                                  'estimation':{'std_dev': float('nan'), 'time': float('nan')},
                                  'object_name': 'det1', 'time': {'start': 20, 'stop': 21.05}},
                                 {'action': 'stage', 
                                  'estimation':{'std_dev': float('nan'), 'time': float('nan')},
                                  'object_name': 'det1', 'time': {'start': 22, 'stop': 22.95}},
                                 {'action': 'unstage', 
                                  'estimation':{'std_dev': float('nan'), 'time': float('nan')},
                                  'object_name': 'det1', 'time': {'start': 20, 'stop': 21.05}},
                                 {'action': 'unstage', 
                                  'estimation':{'std_dev': float('nan'), 'time': float('nan')},
                                  'object_name': 'det1', 'time': {'start': 22, 'stop': 22.95}}])
                                                                #add 2 'trigger' values for det1
        

def test_EpicsMotorEstTime(setup_telemetry, RE, hw):
    #test the time estimation for epics motorlike devices.
    motor1 = hw.motor1
    err_msg = 'calculated set_time/std_dev differs from expected value'
    numpy.testing.assert_almost_equal(motor1.est_time.set(1,2,1,4).est_time, 4.9975, decimal = 4) 
    numpy.testing.assert_almost_equal(motor1.est_time.set(1,2,1,4).std_dev, 0.0759, decimal = 4)
    numpy.testing.assert_almost_equal(motor1.est_time.stage().est_time, 1.0000, decimal = 4)
    numpy.testing.assert_almost_equal(motor1.est_time.stage().std_dev, 0.0707, decimal = 4)
    numpy.testing.assert_almost_equal(motor1.est_time.unstage().est_time, 1.0000, decimal = 4)
    numpy.testing.assert_almost_equal(motor1.est_time.unstage().std_dev, 0.0707, decimal = 4)

def test_ADEstTime(setup_telemetry, RE, hw):
    det1 = hw.det1
    numpy.testing.assert_almost_equal(det1.est_time.trigger(1,2,'fixed',1,4).est_time, 5.0000, decimal = 4)
    numpy.testing.assert_almost_equal(det1.est_time.trigger(1,2,'fixed',1,4).std_dev, 0.0707, decimal = 4)
    numpy.testing.assert_almost_equal(det1.est_time.stage().est_time, 1.0000, decimal = 4)
    numpy.testing.assert_almost_equal(det1.est_time.stage().std_dev, 0.0707, decimal = 4)
    numpy.testing.assert_almost_equal(det1.est_time.unstage().est_time, 1.0000, decimal = 4)
    numpy.testing.assert_almost_equal(det1.est_time.unstage().std_dev, 0.0707, decimal = 4)
