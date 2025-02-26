import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple
from typing import Optional, Tuple, List

pygame.init()

try:
    font = pygame.font.Font('arial.ttf', 25)
except FileNotFoundError:
    print("No se encontró 'arial.ttf'. Usando fuente predeterminada.")
    font = pygame.font.SysFont('arial', 25)

# Constantes de colores RGB
WHITE = (255, 255, 255)
RED = (220, 20, 60)
BLUE1 = (30, 144, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Configuración del juego
BLOCK_SIZE = 20
SPEED = 90

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class SnakeGameAI:
    def __init__(self, width: int = 640, height: int = 480, n_game: int = 0, record: int = 0) -> None:
        self.width: int = width
        self.height: int = height
        self.n_game: int = n_game
        self.record: int = record
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Training Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        
        self.reward_history = []

    def reset(self):
        """Reinicia el estado del juego."""
        self.steps: int = 0
        self.direction: Direction = random.choice(list(Direction))
        self.head: Point = Point(self.width // 2, self.height // 2)
        self.snake: List[Point] = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        self.score: int = 0
        self.food: Optional[Point] = None
        self._place_food()
        self.frame_iteration: int = 0
        
        self.reward_history = []

    def _place_food(self):
        """Coloca comida en una posición aleatoria."""
        while True:
            x: int = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y: int = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def play_step(self, action: List[int], n_game: int, record: int) -> Tuple[int, bool, int]:
        """Ejecuta un paso del juego y retorna (reward, game_over, score)."""
        prev_distance: int = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
    
        self.steps += 1
        self.n_game = n_game
        self.record = record
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)

        reward: int = 0
        game_over: bool = False
        
        if self.is_collision() or self.frame_iteration > 90 * len(self.snake):
            game_over = True
            reward = -10
            self.reward_history.append(reward)
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self.reward_history.append(reward)
            self._place_food()
        else:
            new_distance: int = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            if new_distance < prev_distance:
                reward = 1
            else:
                reward = -1
                self.reward_history.append(reward)
            self.snake.pop()
            
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, point: Optional[Point] = None) -> bool:
        """Verifica si hay colisión."""
        if point is None:
            point = self.head
        if (point.x >= self.width or point.x < 0 or point.y >= self.height or point.y < 0):
            return True
        if point in self.snake[1:]:
            return True
        return False

    def _update_ui(self) -> None:
        """Actualiza la interfaz gráfica."""
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        score_text = font.render(f"Score: {self.score}", True, WHITE)
        n_game_text = font.render(f"Game: {self.n_game}", True, WHITE)
        record_text = font.render(f"Record: {self.record}", True, WHITE)
        self.display.blit(score_text, [0, 0])
        self.display.blit(n_game_text, [0, 30])
        self.display.blit(record_text, [0, 60])
        pygame.display.flip()

    def _move(self, action: List[int]) -> None:
        """
        Mueve la serpiente según la acción dada.
        Acción: [recto, derecha, izquierda]
        """
        directions: List[Direction] = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx: int = directions.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir: Direction = directions[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir: Direction = directions[(idx + 1) % 4]
        else:
            new_dir: Direction = directions[(idx - 1) % 4]

        self.direction = new_dir

        x: int = self.head.x
        y: int = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)