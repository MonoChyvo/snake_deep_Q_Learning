import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Configuración de Pygame
pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Colores RGB
WHITE = (255, 255, 255)
RED = (220, 20, 60)
BLUE1 = (30, 144, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Configuración del Juego
BLOCK_SIZE = 20
SPEED = 120

class SnakeGameAI:
    
    def __init__(self, width=640, height=480, n_game=0, record=0):
        self.width = width
        self.height = height
        self.n_game = n_game
        self.record = record
        
        # Inicializar ventana del juego
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self):
        # Inicializar el estado del juego
        self.direction = Direction.RIGHT
        
        # Inicializar la serpiente
        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        
        # Inicializar puntaje y comida
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        # Colocar comida en una posición aleatoria
        while True:
            x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                break
        
    def play_step(self, action, n_game, record):
        self.n_game = n_game
        self.record = record
        self.frame_iteration += 1
        
        # Procesar eventos de salida
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Mover la serpiente
        self._move(action)
        self.snake.insert(0, self.head)
        
        # Verificar colisiones
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 90 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # Verificar si come comida
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # Actualizar UI y reloj
        self._update_ui()
        self.clock.tick(SPEED)
        
        return reward, game_over, self.score
    
    def is_collision(self, point=None):
        # Verificar colisión con los límites o con el cuerpo de la serpiente
        if point is None:
            point = self.head
        
        if (point.x >= self.width or point.x < 0 or
            point.y >= self.height or point.y < 0):
            return True
        
        if point in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        # Dibujar fondo y elementos del juego
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Mostrar datos
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        n_game_text = font.render(f"Game: {self.n_game}", True, WHITE)
        record_text = font.render(f"Record: {self.record}", True, WHITE)

        self.display.blit(score_text, [0, 0])
        self.display.blit(n_game_text, [0, 30])
        self.display.blit(record_text, [0, 60])
        
        pygame.display.flip()
        
    def _move(self, action):
        # Definir direcciones en sentido horario
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = directions.index(self.direction)
        
        # Actualizar dirección según la acción
        if np.array_equal(action, [1, 0, 0]):
            new_dir = directions[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = directions[(idx + 1) % 4]
        else:
            new_dir = directions[(idx - 1) % 4]
        
        self.direction = new_dir
        
        # Actualizar la posición de la cabeza
        x, y = self.head
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        
        self.head = Point(x, y)
