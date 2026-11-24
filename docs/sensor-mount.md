# Integrated PCB robot: geometry revision integrated-pcb-1

The blue reference photograph motivates the continuous rounded body, rear
drive wheels, central neck and wide curved sensor PCB. No brand artwork, LCD,
battery replica or disconnected electronics is included. This is a nominal
mechanical concept suitable for simulation, not fabrication-ready PCB/CAD or
measured hardware. Actual manufacturing still needs component selection,
mounting-hole design, structural checks and electrical layout.

## Coordinates and dimensions

The axle midpoint is `(0,0)`; +X is forward, +Y left, +Z up. Positive equal motor
commands move toward the sensor bar. Index 0 is leftmost, 7 central, 14 rightmost.

| Parameter | Current starting configuration |
| --- | ---: |
| Main body length / width | 0.090 / 0.130 m |
| Main body longitudinal bounds | -0.030 to +0.060 m |
| PCB thickness / nominal center height | 0.0025 / 0.009 m |
| Wheel spacing / radius / width | 0.175 / 0.0165 / 0.012 m |
| Total width over tires | 0.187 m |
| Passive caster X / ball radius | +0.045 / 0.006 m |
| Total nominal mass | 0.550 kg |
| Fixed assembly / each wheel / caster masses | 0.490 / 0.025 / 0.010 kg |
| Fixed-assembly COM X / nominal Z | +0.018 / 0.021 m |
| Effective inertia envelope height | 0.025 m |
| Sensors | 15 binary, pose-dependent downward apertures |
| Center sensor X (`sensor_offset`) | 0.090 m |
| Circular arc radius / span | 0.110 m / 1.70 rad |
| Outer sensor X / lateral Y | 0.052598146 m / ±0.082640845 m |
| Outer sensor planar axle distance | 0.097959554 m |
| Center-to-center lateral sensor span | 0.165281689 m |
| Adjacent angular spacing | 1.70 / 14 rad |
| Curved PCB radial width | 0.016 m |
| Neck width | 0.028 m |
| Exposed neck X bounds | +0.060 to approximately +0.082 m |
| Nominal sensor aperture height | 0.006 m |

For `i=0,...,14`, with `a=sensor_angle`, `R=sensor_radius`, `d=sensor_offset`:

```text
theta_i = -a/2 + i*a/14
x_i = d - R + R*cos(theta_i)
y_i = -R*sin(theta_i)
```

The name semicircle denotes the curved-layout preset; this practical arc is
1.70 rad, not a complete pi-radian half circle. Changing only `sensor_offset`
translates the curved board, all 15 packages and sites together and extends
the integral neck. Main body, motor brackets, wheels/caster, sensor spacing,
mass, effective inertia and motor limits remain fixed. Too-short necks or
wheel interference raise errors instead of silently changing dimensions.

The body and neck use tangent-continuous sampled curves; the entire visible
PCB is one watertight extruded concave perimeter. Invisible convex collision
strips preserve the openings between body and wings. Motor/gearbox axes are
coaxial with the tires, brackets seat on the PCB, and rotating axle shafts
reach the gearboxes. A fixed flanged cup holds the independently rolling ball.
Sensor packages touch the board underside; their aperture locations and the
actual reflectance sites are identical. Sensor packages/pads are visual
details; the rigid PCB, wheel, bracket and caster surfaces provide contacts.

The fixed 490 g mass is a lumped assembly property including unmodeled onboard
hardware. It is not assigned as the material density of a bare thin PCB. Its
equivalent-box inertia uses the main-body length/width and 25 mm envelope
height. Translating the light sensor mount holds this approximation constant;
this isolates sensing geometry and does not model added PCB inertia/flexibility.

## Methodological changes

The prior development model used a 150 mm inertia envelope, a wider
120 mm / 2.10 rad sensor arc, 80 mm center offset, a rear caster at -75 mm,
and COM X=-25 mm. The redesigned body is 90 mm long; rear wheel placement
requires a forward caster and COM inside the new support triangle. Both
mechanical dynamics and observations therefore changed. Earlier model/run
artifacts cannot be treated as equivalent evidence. The `design_revision` and
full configurations distinguish them; no main-study outcome is carried over.

The original paper geometry remains separately available with `--preset paper`:
50 mm radius, 50 mm center offset and 1.57 rad span. Its historical XLINE layout
is available by changing that preset's `sensor_layout`; XLINE is not mounted on
the curved competition PCB. Historical pixel conversions are not calibration.

At 140 mm center offset the outer sensors are 0.102598146 m ahead. That adds
50 mm of longitudinal preview to every sensor relative to the current model.
For straight aligned motion, a rough center preview time is offset/speed;
it is not a corner prediction model. Earlier detection can also trigger an
early turn, move the axle away from the intended vertex, or miss ordered gates.
The PID pilot compares these effects with fixed shared gains and physics.

## Research freeze and validation

Current JSON defaults are a recommendation, not a finalized geometry. The user
selects a configuration with `freeze-robot`; the recorded configuration/MJCF
hashes are then required by main training. No freeze was created automatically.

Physical checks cover stable three-point support and weight balance, brackets
seated on the board, shaft/gearbox continuity, straight/reverse/braking/pivots,
bounded torque/finite acceleration, all 15 optical locations under a moved and
rotated pose, and floor texture/sensor orientation. Board clearance is measured
under nominal motion; the pilot logs 60 Hz tilt and minimum PCB vertex height.
These do not establish impact, fault, degradation or external-disturbance robustness.

All motor limits, controller algorithms, physics/control rates and route
evaluation rules were retained during the offset comparison. PID gains were
tuned once explicitly for the pilot and saved separately; default historical
PID gains were not overwritten.
