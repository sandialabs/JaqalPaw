To clone the codebase, make sure to add the `--recurse` flag to the clone operation:

```
git clone --recurse git@gitlab.sandia.gov:/jaqal/jaqalpaw.git JaqalPaw/
```

If you fail to add the recurse flag initially, you can subsequently clone the submodule by running

```
git submodule update --init
```

to populate the Jaqal directory.

# Running The Emulator

For the initial version, simply run the top-level file `main.py`. This method bypasses some of the preprocessor steps for identifying the target gate pulse definition class, so this must be inserted manually and an instance of the class is directly imported and passed in via the `pulse_definition` keyword argument.

The default code uses the gate pulse definitions in the top-level `PulseDefinitions.py` file, and executes the gates specified in the top-level `test.jql` Jaqal file. Thus, modifications to the code should be reflected either in these files, or new files that are subsequently imported/identified in `main.py`.

# Pulse Specification

Pulses are specified on a per-channel basis using the `PulseData` class, which has the following call signature:

```python
PulseData(channel, 
          duration,
          amp0=0,
          amp1=0,
          freq0=0,
          freq1=0,
          phase0=0,
          phase1=0,
          framerot0=0,
          framerot1=0,
          enable_mask=0,
          fb_enable_mask=0,
          sync_mask=0,
          apply_at_eof_mask=0,
          clr_frame_rot_mask=0,
          waittrig=False
         )
```

| parameter                                            | allowed types                                  | description                                                  |
| ---------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| `channel`                                            | `int`                                          | Corresponding channel for which the parameters will be applied. |
| `duration`                                           | `float`                                        | Total duration in seconds for which parameters will be applied |
| `amp0`, `amp1`, `freq0`, `freq1`, `phase0`, `phase1` | `float`, `tuple`, `list`, `Discrete`, `Spline` | Amplitude/frequency/phase applied to tones 0 and 1           |
| `framerot0`, `framerot1`                             | `float`, `tuple`, `list`, `Discrete`, `Spline` | z rotation applied to tones 0 and 1. Frame rotation was used as a keyword argument since the rotation in z is specific to the experimental hardware and nomenclature. This is effectively identical to changing the phase, except that the value applied persists and is taken to be the new zero value for subsequent pulses. |
| `xxxx_mask`                                          | `int`                                          | All mask types are treated as two-bit bitmasks, where tone0 and tone0 are controlled by the least and most significant bits respectively. In other words, allowed values are 0-3, where 0b00 has no effect, 0b01 affects tone 0, 0b10 affects tone 1 and 0b11 affects both tones. |
| `enable_mask`                                        | `int`                                          | Toggles the static value of the output enable. When the output enable is disabled, or zero, the resulting output of the tone is ignored and forced to zero. |
| `fb_enable_mask`                                       | `int`                                          | Toggles the static value of the frequency feedback  enable.  |
| `sync_mask`                                          | `int`                                          | Applies a global synchronization at the beginning of the pulse so that the phase is aligned to the global clock for the given frequency. |
| `apply_at_eof_mask`                                  | `int`                                          | Applies `framerot` parameters with the _next_ pulse. This is to account for AC Stark shifts resulting from compensated pulses that take compensate for the frequency shift during the pulse. |
| `clr_frame_rot_mask`                                 | `int`                                          | Resets the accumulated phase from previous `framerot` calls to 0 |
| `waittrig`                                           | `bool`                                         | waits for external trigger (hardware or software) before applying pulse |

# Gate Pulse Definition File

Gates are defined at the pulse level in a python format that is referenced by the Jaqal file in the form of 

```
from GatePulseFileName.GatePulseClassName usepulses *
```

with optional relative  imports from subdirectories such as

```
from SubDirectory1.SubDirectory2.GatePulseFileName.GatePulseClassName usepulses *
```

Class level definitions with type annotations are exposed to the main experimental control software, and typically provide hooks that allow us to override the variable's final value to perform scans or pass in the latest calibrated values. The annotation type currently has no effect, and thus can be any native python type, e.g. 

```python
class GatePulses:
    pulse_duration : float = 2e-6
    global_freq : float = 200.0
    individual_freq : float = 242.0
```

Gates are defined as member functions and have `gate_` prefix before the name of the gate. In other words, a gate named `R` in Jaqal will be defined as `gate_R` in the gate definition class. Helper functions referenced by the class must be member functions of the class and don't need the `gate_` prefix. 

The input argument signature of a gate corresponds to the arguments that can be passed in from the Jaqal file. Default arguments are allowed, but cannot be referenced as keyword arguments in Jaqal. For example

```python
class GatePulses:
    pulse_duration : float = 2e-6
    global_freq : float = 200.0
    individual_freq : float = 242.0
        
    def gate_R(self, channel, theta=0):
    	...
```

can be called in Jaqal as

```
// Example header and declarations
from GatePulseFileName.GatePulses usepulses *
let pi 3.14159265
register q[8]

// Call to gate with input arguments
R q[1] pi
R q[2]
```

## Return Signature of Gate Definitions

Each gate definition function must return a list of `PulseData` objects. Objects targeting the same channel are applied serially in the order received, and objects targeting different channels will be run in parallel. Serial/parallel objects can be combined into the same list. By convention, the global beam is defined as channel 0.

```python
global_beam_channel = 0

class GatePulses:
    pulse_duration : float = 2e-6
    global_freq : float = 200.0
    individual_freq : float = 242.0
        
    def gate_H(self, channel, theta):
        "Example counter-propagating Hadamard gate"
        return [PulseData(global_beam_channel, 
                          self.pulse_duration*3,
                          amp0=100,
                          freq0=self.global_freq),
                PulseData(channel, 
                          self.pulse_duration, 
                          amp0=100,
                          phase0=theta,
                          freq0=self.individual_freq),
                PulseData(channel, 
                          self.pulse_duration*2, 
                          amp0=100,
                          phase0=theta+90, 
                          freq0=self.individual_freq)]
```

Gates can call other gates, if desired, as long as the return signature is still a list

```python
global_beam_channel = 0

class GatePulses:
    pulse_duration : float = 2e-6
    global_freq : float = 200.0
    individual_freq : float = 242.0
        
    def gate_R(self, channel, theta, duration_scale=1):
        "Example counter-propagating Hadamard gate"
        return [PulseData(channel, 
                          self.pulse_duration*duration_scale, 
                          amp0=100,
                          phase0=theta,
                          freq0=self.individual_freq)]
    
    def gate_I(self, channel, duration):
        return [PulseData(channel, 
                          duration, 
                          amp0=100,
                          phase0=theta,
                          freq0=self.individual_freq)]
    
    def gate_RamseyEcho(self, channel, duration, analyze_phase=0):
        return self.gate_R(channel, 0) + \
    		   self.gate_I(channel, duration) + \
               self.gate_R(channel, 0, 2) + \
               self.gate_I(channel, duration) + \
               self.gate_R(channel, analyze_phase)
```

thus supporting an equivalency between the following Jaqal implementations:

```
let analyze_phase 180 // degrees
let ramsey_wait 1 // seconds
register q[8]

macro ramsey targ waittime{
	R targ 0
	I waittime
	R targ 0 2
	I waittime
	R targ analyze_phase
}

// Implementation 1 (macro)
ramsey q[1] ramsey_wait

// Implementation 2 (gate)
RamseyEcho q[1] ramsey_wait

// Implementation 3 (direct)
R q[1] 0
I ramsey_wait
R q[1] 0 2
I ramsey_wait
R q[1] analyze_phase
```

