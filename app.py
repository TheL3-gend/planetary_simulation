import pygame
import random
import math
import sys

# Window dimensions
WIDTH, HEIGHT = 800, 600

# Simulation parameters
NUM_BODIES = 30
G = 6.674e-3       # Gravitational constant (scaled for visualization)
DT = 0.1           # Time step

class Body:
    def __init__(self, x, y, vx, vy, mass):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass
        # radius for drawing (proportional to mass)
        self.radius = max(2, int(math.sqrt(self.mass) * 0.5))

    def compute_force(self, others):
        fx = fy = 0.0
        for other in others:
            if other is self:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            dist_sq = dx*dx + dy*dy
            if dist_sq == 0:
                continue
            dist = math.sqrt(dist_sq)
            # Newton's law of universal gravitation
            force = G * self.mass * other.mass / dist_sq
            fx += force * dx / dist
            fy += force * dy / dist
        return fx, fy

    def update(self, bodies):
        fx, fy = self.compute_force(bodies)
        # acceleration
        ax = fx / self.mass
        ay = fy / self.mass
        # integrate velocity
        self.vx += ax * DT
        self.vy += ay * DT
        # integrate position
        self.x += self.vx * DT
        self.y += self.vy * DT

    def draw(self, screen):
        color = (255, 255, 255)
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.radius)


def create_bodies():
    bodies = []
    for _ in range(NUM_BODIES):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        vx = random.uniform(-1, 1)
        vy = random.uniform(-1, 1)
        mass = random.uniform(5, 50)
        bodies.append(Body(x, y, vx, vy, mass))
    return bodies


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Gravity Simulation")
    clock = pygame.time.Clock()

    bodies = create_bodies()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Update all bodies
        for body in bodies:
            body.update(bodies)

        # Draw
        screen.fill((0, 0, 0))
        for body in bodies:
            body.draw(screen)
        pygame.display.flip()

        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
