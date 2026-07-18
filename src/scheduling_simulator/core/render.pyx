# renderer.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import pygame
from .cluster cimport Observation

cimport cython

cdef class Renderer:
    cdef object screen
    cdef object font
    cdef object small_font
    cdef int width, height, cell_size, margin

    def __init__(self, int width=1700, int height=800, int cell_size=24):
        pygame.init()
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.margin = 20
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Environment Renderer")
        self.font = pygame.font.SysFont("Consolas", 18)
        self.small_font = pygame.font.SysFont("Consolas", 12)

    cpdef close(self):
        pygame.quit()

    cdef tuple _status_color(self, int status):
        # Adjust these numbers if your status enum is different
        if status == 0:       # not_created
            return (255, 0, 0)       # red
        elif status == 1:     # pending
            return (0, 255, 0)       # green
        elif status == 2:     # running
            return (0, 255, 0)       # green
        elif status == 3:     # completed
            return (150, 150, 150)   # grey

        return (255, 255, 255)

    cdef tuple _job_color(self, int job):
        if job <= 0:
            return (50,50,50)
        return ((job*67)%255,(job*131)%255,(job*199)%255)

    cpdef render(self, Observation obs):
        cdef:
            int machine,row,col,job,value
            int machines,rows,cols,jobs
            int x,y
            int job_rows,job_cols
            int machines_per_row=2
            int jobs_per_row=4
            int block_w,block_h
            tuple rect,color
            object text,text_rect

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.screen.fill((30,30,30))
        text=self.font.render(f"Simulation Time : {obs.time}",True,(255,255,255))
        self.screen.blit(text,(20,10))

        # Machines (left)
        machines=obs.machines_usage.shape[0]
        rows=obs.machines_usage.shape[1]
        cols=obs.machines_usage.shape[2]
        block_w=cols*self.cell_size+80
        block_h=rows*self.cell_size+55

        for machine in range(machines):
            x=20+(machine%machines_per_row)*block_w
            y=50+(machine//machines_per_row)*block_h
            self.screen.blit(self.font.render(f"Machine {machine}",True,(255,255,0)),(x,y))
            for row in range(rows):
                for col in range(cols):
                    value=obs.machines_capacity[machine,row,col] - obs.machines_usage[machine,row,col]
                    rect=(x+col*self.cell_size,y+22+row*self.cell_size,self.cell_size-1,self.cell_size-1)
                    pygame.draw.rect(self.screen,self._job_color(value),rect)
                    text=self.small_font.render(str(value),True,(255,255,255))
                    text_rect=text.get_rect(center=(rect[0]+self.cell_size//2,rect[1]+self.cell_size//2))
                    self.screen.blit(text,text_rect)

        # Job usage (right)
        # Job usage (right)

        jobs = obs.status.shape[0]

        job_rows = obs.jobs_usage.shape[1]
        job_cols = obs.jobs_usage.shape[2]

        # Available area on right side
        x0 = self.width//2 + 40
        available_w = self.width - x0 - 20

        # Calculate jobs per row dynamically
        block_w = job_cols*self.cell_size + 70
        jobs_per_row = max(1, available_w // block_w)

        block_h = job_rows*self.cell_size + 55

        self.screen.blit(
            self.font.render("Job Usage", True, (255,255,255)),
            (x0,20)
        )

        for job in range(jobs):

            x = x0 + (job % jobs_per_row) * block_w
            y = 50 + (job // jobs_per_row) * block_h

            self.screen.blit(
                self.font.render(
                    f"Job {job}",
                    True,
                    self._status_color(obs.status[job])
                ),
                (x,y)
            )

            for row in range(job_rows):
                for col in range(job_cols):

                    value = obs.jobs_usage[job,row,col]

                    color = (
                        self._job_color(job)
                        if value
                        else (60,60,60)
                    )

                    rect = (
                        x + col*self.cell_size,
                        y + 22 + row*self.cell_size,
                        self.cell_size-1,
                        self.cell_size-1
                    )

                    pygame.draw.rect(
                        self.screen,
                        color,
                        rect
                    )

                    text = self.small_font.render(
                        str(value),
                        True,
                        (255,255,255)
                    )

                    text_rect = text.get_rect(
                        center=(
                            rect[0]+self.cell_size//2,
                            rect[1]+self.cell_size//2
                        )
                    )

                    self.screen.blit(text,text_rect)
        # Job info bottom
        # Job info panel

        x = self.width//2 + 40
        y = self.height - 160

        self.screen.blit(
            self.font.render("Jobs", True, (255,255,255)),
            (x,y)
        )

        y += 24

        max_lines = 6   # prevents going below screen

        for job in range(min(jobs, max_lines)):

            text = self.font.render(
                f"{job:2d} S:{obs.status[job]} TTL:{obs.ttl[job]} "
                f"A:{obs.arrival_time[job]} Size:{obs.size[job]}",
                True,
                self._status_color(obs.status[job])
            )

            self.screen.blit(text,(x,y))

            y += 20

        pygame.display.flip()
