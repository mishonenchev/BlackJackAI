import os
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.error import DependencyNotInstalled


def cmp(a, b):
    return float(a > b) - float(a < b)


# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
actions = [str]

def draw_card(np_random):
    return int(np_random.choice(deck))


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackEnv(gym.Env):
    """
    Blackjack is a card game where the goal is to beat the dealer by obtaining cards
    that sum to closer to 21 (without going over 21) than the dealers cards.

    ### Description
    Card Values:

    - Face cards (Jack, Queen, King) have a point value of 10.
    - Aces can either count as 11 (called a 'usable ace') or 1.
    - Numerical cards (2-9) have a value equal to their number.

    This game is played with an infinite deck (or with replacement).
    The game starts with the dealer having one face up and one face down card,
    while the player has two face up cards.

    The player can request additional cards (hit, action=1) until they decide to stop (stick, action=0)
    or exceed 21 (bust, immediate loss).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust, the player wins.
    If neither the player nor the dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.

    ### Action Space
    There are two actions: stick (0), and hit (1).

    ### Observation Space
    The observation consists of a 3-tuple containing: the player's current sum,
    the value of the dealer's one showing card (1-10 where 1 is ace),
    and whether the player holds a usable ace (0 or 1).

    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto (http://incompleteideas.net/book/the-book-2nd.html).

    ### Rewards
    - win game: +1
    - lose game: -1
    - draw game: 0
    - win game with natural blackjack:

        +1.5 (if <a href="#nat">natural</a> is True)

        +1 (if <a href="#nat">natural</a> is False)

    ### Arguments

    ```
    gym.make('Blackjack-v1', natural=False, sab=False)
    ```

    <a id="nat">`natural=False`</a>: Whether to give an additional reward for
    starting with a natural blackjack, i.e. starting with an ace and ten (sum is 21).

    <a id="sab">`sab=False`</a>: Whether to follow the exact rules outlined in the book by
    Sutton and Barto. If `sab` is `True`, the keyword argument `natural` will be ignored.
    If the player achieves a natural blackjack and the dealer does not, the player
    will win (i.e. get a reward of +1). The reverse rule does not apply.
    If both the player and the dealer get a natural, it will be a draw (i.e. reward 0).

    ### Version History
    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None, natural=False, sab=False):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2))
        )

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural

        # Flag for full agreement with the (Sutton and Barto, 2018) definition. Overrides self.natural
        self.sab = sab

        self.render_mode = render_mode

    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                self.actions.append(f"Player hit {self.player[-1]} and busted")
                terminated = True
                reward = -1.0
            else:
                self.actions.append(f"Player hit {self.player[-1]}")
                terminated = False
                reward = 0.0
        else:  # stick: play out the dealers hand, and score
            terminated = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            if score(self.player) > score(self.dealer):
                self.actions.append('Player stand and won')
                reward = 1.0
            else:
                self.actions.append('Player stand and lost')
                reward = -1.0
            if self.sab and is_natural(self.player) and not is_natural(self.dealer):
                # Player automatically wins. Rules consistent with S&B
                reward = 1.0
            elif (
                not self.sab
                and self.natural
                and is_natural(self.player)
                and reward == 1.0
            ):
                # Natural gives extra points, but doesn't autowin. Legacy implementation
                reward = 1.5

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.actions = []
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)

        _, dealer_card_value, _ = self._get_obs()

        suits = ["C", "D", "H", "S"]
        self.dealer_top_card_suit = self.np_random.choice(suits)
        self.dealer_hidden_card_suit = self.np_random.choice(suits)

        if dealer_card_value == 1:
            self.dealer_top_card_value_str = "A"
        elif dealer_card_value == 10:
            self.dealer_top_card_value_str = self.np_random.choice(["J", "Q", "K"])
        else:
            self.dealer_top_card_value_str = str(dealer_card_value)
        
        if self.dealer[1] == 1:
            self.dealer_hidden_card_value_str = "A"
        elif self.dealer[1] == 10:
            self.dealer_hidden_card_value_str = self.np_random.choice(["J", "Q", "K"])
        else:
            self.dealer_hidden_card_value_str = str(self.dealer[1])

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )

        player_sum, dealer_card_value, usable_ace = self._get_obs()
        screen_width, screen_height = 1000, 750
        card_img_height = screen_height // 5
        card_img_width = int(card_img_height * 142 / 197)
        action_bottom_rect = 0
        spacing = screen_height // 20
        num_cards_dealer = len(self.dealer)
        num_cards_player = len(self.player)
        suits = ["C", "D", "H", "S"]

        bg_color = (7, 99, 36)
        white = (255, 255, 255)

        if not hasattr(self, "screen"):
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                pygame.font.init()
                self.screen = pygame.Surface((screen_width, screen_height))

        if not hasattr(self, "clock"):
            self.clock = pygame.time.Clock()

        self.screen.fill(bg_color)

        def get_image(path):
            cwd = os.path.dirname(__file__)
            image = pygame.image.load(os.path.join(cwd, path))
            return image

        def get_font(path, size):
            cwd = os.path.dirname(__file__)
            font = pygame.font.Font(os.path.join(cwd, path), size)
            return font

        small_font = get_font(
            os.path.join("font", "Minecraft.ttf"), screen_height // 20
        )
        action_font = get_font(
            os.path.join("font", "times.ttf"), screen_height // 30
        )
        dealer_text = small_font.render(
            "Dealer: " + str(sum_hand(self.dealer)), True, white
        )
        dealer_text_rect = self.screen.blit(dealer_text, (spacing, spacing))

        def scale_card_img(card_img):
            return pygame.transform.scale(card_img, (card_img_width, card_img_height))

        dealer_card_img = scale_card_img(
            get_image(
                os.path.join(
                    "img",
                    f"{self.dealer_top_card_suit}{self.dealer_top_card_value_str}.png",
                )
            )
        )
        dealer_card_rect = self.screen.blit(
            dealer_card_img,
            (
                screen_width // num_cards_dealer - card_img_width - spacing // num_cards_dealer,
                dealer_text_rect.bottom + spacing,
            ),
        )

        hidden_card_img = scale_card_img(
            get_image(
                os.path.join(
                    "img",
                    f"{self.dealer_hidden_card_suit}{self.dealer_hidden_card_value_str}.png",
                )
            )
        )
        self.screen.blit(
            hidden_card_img,
            (
                screen_width // num_cards_dealer + spacing // num_cards_dealer,
                dealer_text_rect.bottom + spacing,
            ),
        )
        if len(self.dealer) >= 3:
            for i in range(2, len(self.dealer)):
                
                dealer_draw_card_suit = self.np_random.choice(suits)
                if self.dealer[i] == 1:
                    dealer_draw_card = "A"
                elif self.dealer[i] == 10:
                    dealer_draw_card = self.np_random.choice(["J", "Q", "K"])
                else:
                    dealer_draw_card = str(self.dealer[i])
                
                aditional_card_img = scale_card_img(
                    get_image(
                        os.path.join(
                            "img",
                            f"{dealer_draw_card_suit}{dealer_draw_card}.png",
                        )
                    )
                )
                self.screen.blit(
                    aditional_card_img,
                    (
                        screen_width // num_cards_dealer + (i - 1) * (card_img_width + spacing),
                        dealer_text_rect.bottom + spacing,
                    ),
                )
            

        player_text = small_font.render(f"Player: {str(sum_hand(self.player))}", True, white)
        player_text_rect = self.screen.blit(
            player_text, (spacing, dealer_card_rect.bottom + 1.5 * spacing)
        )

        for i in range(len(self.player)):
            
            player_draw_card_suit = self.np_random.choice(suits)
            if self.player[i] == 1:
                player_draw_card = "A"
            elif self.player[i] == 10:
                player_draw_card = self.np_random.choice(["J", "Q", "K"])
            else:
                player_draw_card = str(self.player[i])
            
            aditional_card_img = scale_card_img(
                get_image(
                    os.path.join(
                        "img",
                        f"{player_draw_card_suit}{player_draw_card}.png",
                    )
                )
            )
            aditional_card_rect = self.screen.blit(
                aditional_card_img,
                (
                    screen_width // num_cards_player + (i - 1) * (card_img_width + spacing) + spacing // num_cards_player,
                    player_text_rect.bottom + spacing,
                ),
            )
            
        action_bottom_rect = aditional_card_rect.bottom
        
        if usable_ace:
            usable_ace_text = action_font.render("usable ace", True, white)
            usable_ace_rect = self.screen.blit(
                usable_ace_text,
                (
                    screen_width // 2 - usable_ace_text.get_width() // 2,
                    aditional_card_rect.bottom + spacing // 2,
                ),
            )
            action_bottom_rect = usable_ace_rect.bottom
        
        for i in range(len(self.actions)):
            player_action_text = action_font.render(self.actions[i], True, white)
            player_action_rect = self.screen.blit(
                player_action_text,
                (
                    screen_width // 2 - player_action_text.get_width() // 2,
                    action_bottom_rect + spacing // 3
                )
            )
            action_bottom_rect = player_action_rect.bottom
        

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if hasattr(self, "screen"):
            import pygame

            pygame.display.quit()
            pygame.quit()


# Pixel art from Mariia Khmelnytska (https://www.123rf.com/photo_104453049_stock-vector-pixel-art-playing-cards-standart-deck-vector-set.html)
