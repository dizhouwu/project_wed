# Project Wed - Scene 2: The Rewind

"Memory is not a static file; it is a reconstruction."

This project is a visual experiment in temporal travel. Using Python as the lens, Scene 2 simulates the psychological acceleration of looking back—where the present blurs, and the past pulls you in with an inescapable velocity.

## 🎞 Cinematic Mechanics

* **Temporal Acceleration**: The sequence follows a non-linear power-law curve ($\Gamma=1.5$). We begin with the stillness of the present (0.2s), gradually surrendering to a frantic rewind (0.04s) as the gravity of the past takes over.
* **Kinetic Turbulence**: To capture the "motion sickness" of time travel, the engine injects randomized spatial shaking and dynamic motion blur. As the rewind speeds up, the visual stability decays—mirroring the disorientation of a mind racing backward.
* **The Overload Point (Flicker)**: At the 70% mark, the system hits a threshold. A high-intensity, distorted frame is injected to simulate a "Memory Flicker." It is the exact moment where logic fails to contain the surge of raw memory—a visual heartbeat in a digital world.
* **Hardware Sync**: Optimized for the intimate screen. Rendered in YUV420P to ensure the light and shadow of these moments translate perfectly to the palm of your hand.

## Tech Stack

* **OpenCV**: Core rendering and image manipulation engine.
* **MoviePy**: Sequence assembly and video container synthesis.
* **NumPy**: Temporal curve logic.
