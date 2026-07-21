from typing import Optional, Tuple
import numpy as np

from scheduling_simulator.core.cluster import Observation
from scheduling_simulator.core.render import DefulatConfiguration

class Configuration:

    def __init__(
        self,
        width: int,
        height: int,
        cell_size: int ,
        margin: int,
        title_under_pedding: int,
        margin_between_machines: int,
        pedding_top: int,
        cell_border_thickness: int,
        primary_title_font_size: int,
        secondary_title_font_size: int,
        job_border: int,
        main_text_color: Tuple[int, int, int],
        secondary_text_color: Tuple[int, int, int],
        bg_color: Tuple[int, int, int],
    ) -> None: ...



class Renderer:
    def __init__(self, to_screen: bool = True, config: Configuration = DefulatConfiguration)
    def render(self, obs: Observation) -> Optional[np.ndarray]
    def close(self) -> None
