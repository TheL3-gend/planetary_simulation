# High-Resolution OpenGL Gravity Simulation

A physically accurate gravity simulation with realistic planet textures, efficient GPU rendering, and interactive camera controls.

## Features

- **Complete Solar System**: All planets, major moons, and the sun with accurate masses, distances, and orbital velocities
- **Accurate Physics**: Uses Newton's law of universal gravitation for realistic orbital mechanics
- **High-Resolution Graphics**: Detailed planet textures and smooth rendering
- **GPU-Accelerated Rendering**: Modern OpenGL with shaders for efficient rendering
- **CPU-Based Physics**: Separate CPU calculations from GPU rendering for best performance
- **Planet Rings**: Saturn, Jupiter, Uranus, and Neptune have realistic ring systems
- **Orbital Trails**: Visualize the paths of celestial bodies
- **Interactive Camera**: Rotate, zoom, and pan with intuitive mouse controls
- **Body Information**: Detailed information about each celestial body
- **Collision Handling**: Bodies can collide and merge with conservation of momentum

## Requirements

- Python 3.6+
- PyGame
- PyOpenGL
- NumPy

## Installation

1. Install the required packages:

```bash
pip install pygame PyOpenGL PyOpenGL_accelerate numpy
```

2. Create the necessary directories:

```bash
mkdir -p textures shaders fonts
```

3. Download planet textures and place them in the textures directory:

You'll need texture files for each planet and moon. Place them in the `textures` directory with filenames matching those in `constants.py`. Example textures:

- sun.jpg
- mercury.jpg
- venus.jpg
- earth.jpg
- mars.jpg
- jupiter.jpg
- saturn.jpg
- uranus.jpg
- neptune.jpg
- moon.jpg
- phobos.jpg
- deimos.jpg
- etc.

You can find planet textures from NASA's image library or other astronomical resources.

## Usage

Run the simulation:

```bash
python main.py
```

### Controls

- **Space**: Pause/Resume simulation
- **Tab**: Cycle through celestial bodies
- **+/-**: Increase/decrease simulation speed
- **T**: Toggle orbital trails
- **L**: Toggle planet labels
- **R**: Reset simulation
- **Mouse Drag**: Rotate camera
- **Mouse Wheel**: Zoom in/out
- **Middle Mouse**: Pan view
- **F**: Toggle free camera mode
- **WASD/QE**: Free camera movement (in free mode)
- **C**: Reset camera
- **Esc**: Exit

## File Structure

- `main.py`: Main entry point and application loop
- `constants.py`: Shared constants and simulation parameters
- `body.py`: Celestial body class with physics properties
- `simulation.py`: Simulation logic and physics calculations
- `renderer.py`: OpenGL rendering and shader management
- `camera.py`: Camera controls and movement
- `ui.py`: User interface elements
- `textures/`: Directory for planet textures
- `shaders/`: Directory for GLSL shaders
- `fonts/`: Directory for UI fonts

## Customization

You can customize the simulation by modifying constants in `constants.py`:

- Adjust `WINDOW_WIDTH` and `WINDOW_HEIGHT` for resolution
- Change `FULLSCREEN` to run in fullscreen mode
- Modify `SCALE_FACTOR` to change visual scaling
- Adjust `SIMULATION_SPEED` to change time acceleration
- Set `SPHERE_DETAIL` to control render quality
- Add or modify planets in the `PLANET_DATA` list

## Physics Model

The simulation uses Newton's law of universal gravitation:

F = G * (m₁ * m₂) / r²

where:
- F is the gravitational force between two bodies
- G is the gravitational constant (6.67430 × 10⁻¹¹ m³ kg⁻¹ s⁻²)
- m₁ and m₂ are the masses of the bodies
- r is the distance between the centers of the bodies

## Extensions

- Add asteroid belt simulation
- Include comets with elliptical orbits
- Add spacecraft with propulsion physics
- Implement relativistic effects for extreme scenarios
- Add star field background
- Include atmospheric effects