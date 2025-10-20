from dataclasses import dataclass


@dataclass
class WargameEnvAction:
    """
    Action structure for the Wargame environment.

    List of ints, where each int contains the action for each wargame model: up (0), down (1), left (2), right (3).
    The length of the list is equal to the number of wargame models.
    """

    actions: list[int]  # Actions for each wargame model
