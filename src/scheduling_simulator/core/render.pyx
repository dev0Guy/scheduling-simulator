# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import pygame
from .cluster cimport Observation
from .job import JobStatus

cimport cython

cdef class Configuration:
    cdef int width, height, cell_size, margin
    cdef int cell_border_thickness
    cdef int primary_title_font_size, secondary_title_font_size
    cdef tuple main_text_color, secondary_text_color, bg_color
    cdef int margin_between_machines,
    cdef int pedding_top, title_under_pedding, job_border


    def __init__(
        self,
        int width,
        int height,
        int cell_size,
        int margin,
        int title_under_pedding,
        int margin_between_machines,
        int pedding_top,
        int cell_border_thickness,
        int primary_title_font_size,
        int secondary_title_font_size,
        int job_border,
        tuple main_text_color,
        tuple secondary_text_color,
        tuple bg_color
    ) -> None:
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.margin = margin
        self.title_under_pedding = title_under_pedding
        self.pedding_top = pedding_top
        self.margin_between_machines = margin_between_machines
        self.primary_title_font_size = primary_title_font_size
        self.secondary_title_font_size = secondary_title_font_size
        self.cell_border_thickness = cell_border_thickness
        self.job_border = job_border
        self.main_text_color = main_text_color
        self.secondary_text_color = secondary_text_color
        self.bg_color = bg_color

cdef Configuration DefulatConfiguration = Configuration(
    width=1400,
    height=700,
    cell_size=25,
    margin = 50,
    title_under_pedding = 5,
    pedding_top = 50,
    margin_between_machines = 30,
    primary_title_font_size = 18,
    secondary_title_font_size = 12,
    job_border = 6,
    cell_border_thickness = 1,
    main_text_color = (255, 255, 255),
    secondary_text_color = (0, 0, 0),
    bg_color = (30, 30, 30)
)

cdef class Renderer:
    cdef object screen
    cdef Configuration config
    cdef object font
    cdef object small_font
    cdef bint to_screen

    def __init__(self, bint to_screen = True, Configuration config = DefulatConfiguration):
        pygame.init()
        self.to_screen = to_screen

        if self.to_screen:
            self.screen = pygame.display.set_mode((self.config.width, self.config.height))
            pygame.display.set_caption("Environment Renderer")
        else:
            self.screen = pygame.Surface((self.config.width, self.config.height))

        self.config = config
        self.screen = pygame.display.set_mode((self.config.width, self.config.height))
        pygame.display.set_caption("Environment Renderer")
        self.font = pygame.font.SysFont("Consolas", self.config.primary_title_font_size)
        self.small_font = pygame.font.SysFont("Consolas", self.config.secondary_title_font_size)

    cpdef close(self):
        pygame.quit()

    cdef void draw_table(
        self,
        int[:, ::1] values,
        unsigned int x,
        unsigned int y,
        unsigned int n_rows,
        unsigned int n_columns,
        object text,
        bint active
    ):
        cdef tuple color
        self.screen.blit(
            self.font.render(text, True, self.config.main_text_color),
            (x, y)
        )
        y = y + self.config.primary_title_font_size + self.config.title_under_pedding
        for row in range(n_rows):
            for col in range(n_columns):
                value = values[row,col]
                cell_horizntal_position = x + col * self.config.cell_size
                cell_virtical_position = y + row * self.config.cell_size
                rect = (
                    cell_horizntal_position,
                    cell_virtical_position,
                    self.config.cell_size - self.config.cell_border_thickness,
                    self.config.cell_size - self.config.cell_border_thickness
                )
                color = self.cell_color(value) if active else (83, 92, 104)
                pygame.draw.rect(self.screen, color , rect)
                text = self.small_font.render(str(value), True, self.config.secondary_text_color)
                text_rect = text.get_rect(
                    center=(rect[0] + self.config.cell_size // 2,
                    rect[1] + self.config.cell_size // 2)
                )
                self.screen.blit(text, text_rect)


    cdef void draw_machines(self, Observation obs, unsigned int start_x, unsigned int start_y, unsigned int width, unsigned int height):
        cdef unsigned int n_machines, n_resources, n_time
        cdef unsigned int y

        n_machines = obs.machines_usage.shape[0]
        n_resources = obs.machines_usage.shape[1]
        n_time = obs.machines_usage.shape[2]

        for machine in range(n_machines):
            y = start_y + (machine * (height + self.config.primary_title_font_size + self.config.margin_between_machines))
            self.draw_table(obs.machines_capacity[machine], start_x, y, n_resources, n_time, f"Machine {machine}:", True)


    cdef void draw_jobs(self, Observation obs, unsigned int start_x, unsigned int start_y, unsigned int width, unsigned int height):

        cdef unsigned int n_jobs, n_resources, n_time
        cdef unsigned int block_width, block_height
        cdef unsigned int jobs_per_row = 2
        cdef unsigned int start_horizntal_position, start_virtical_position

        n_jobs = obs.status.shape[0]
        n_resources = obs.jobs_usage.shape[1]
        n_time = obs.jobs_usage.shape[2]

        for job in range(n_jobs):
            x = start_x + (job % jobs_per_row) * (width + self.config.margin)
            y = start_y + (job // jobs_per_row) * (height + self.config.primary_title_font_size + self.config.margin_between_machines)
            inner_height = n_resources * self.config.cell_size
            rect = (
                x - (self.config.job_border // 2),
                y + self.config.primary_title_font_size + self.config.title_under_pedding - (self.config.job_border // 2),
                width + self.config.job_border,
                height + self.config.job_border
            )
            pygame.draw.rect(self.screen, self._status_color(obs.status[job]), rect)
            self.draw_table(obs.jobs_usage[job], x, y, n_resources, n_time, f"Job {job}:", obs.status[job] == JobStatus.PENDING)

            meta_text = self.small_font.render(
                f"S:{obs.status[job]} TTL:{obs.ttl[job]} "
                f"A:{obs.arrival_time[job]} Size:{obs.size[job]}",
                True,
                self._status_color(obs.status[job])
            )
            meta_y = y + height + self.config.margin_between_machines - (self.config.secondary_title_font_size // 2) + 1
            self.screen.blit(meta_text, (x, meta_y))

    cpdef object render(self, Observation obs):
        cdef:
            unsigned int block_width, block_height
            unsigned int n_resources, n_time
            unsigned int machine_horizntal_position, job_horizntal_position
            object text, event

        if self.to_screen:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

        self.screen.fill(self.config.bg_color)
        text = self.font.render(f"Time : {obs.time}", True, self.config.main_text_color)
        self.screen.blit(text, ((self.config.width - text.get_width()) // 2, 10))

        n_resources = obs.jobs_usage.shape[1]
        n_time = obs.jobs_usage.shape[2]

        block_width = self.config.cell_size * n_time
        block_height = n_resources * self.config.cell_size

        machine_horizntal_position = block_width // 2
        job_horizntal_position =  machine_horizntal_position + block_width + self.config.margin

        self.draw_machines(obs, machine_horizntal_position, self.config.pedding_top, block_width, block_height)
        self.draw_jobs(obs, job_horizntal_position, self.config.pedding_top, block_width, block_height)

        if not self.to_screen:
            return pygame.surfarray.array3d(self.screen).transpose((1, 0, 2))

        pygame.display.flip()


    cdef tuple cell_color(self, int value):
        cdef int red, green

        if value == 0:
            return (83, 92, 104)

        red = 255 - value
        green = value

        return (red, green, 0)

    cdef tuple _status_color(self, int status):
        # Adjust these numbers if your status enum is different
        if status == JobStatus.NOT_CREATED:       # not_created
            return (255, 0, 0)       # red
        elif status == JobStatus.PENDING:     # pending
            return (0, 255, 0)       # green
        elif status == JobStatus.RUNNING:     # running
            return (0, 255, 0)       # green
        elif status == JobStatus.COMPLETED:     # completed
            return (150, 150, 150)   # grey

        return (255, 255, 255)


    cdef tuple _job_border_color(self, int status):
        if status == JobStatus.NOT_CREATED:
            return (255, 0, 0)       # red
        elif status == JobStatus.PENDING:
            return (0, 255, 0)       # green
        elif status == JobStatus.RUNNING:
            return (150, 150, 150)   # grey
        elif status == JobStatus.COMPLETED:
            return (0, 0, 0)         # black

        return (255, 255, 255)
