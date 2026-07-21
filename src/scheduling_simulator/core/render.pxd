from .cluster cimport Observation
from .job import JobStatus

cdef class Configuration:
    cdef int width, height, cell_size, margin
    cdef int cell_border_thickness
    cdef int primary_title_font_size, secondary_title_font_size
    cdef tuple main_text_color, secondary_text_color, bg_color
    cdef int margin_between_machines,
    cdef int pedding_top, title_under_pedding, job_border

cdef class Renderer:
    cdef object screen
    cdef Configuration config
    cdef object font
    cdef object small_font
    cdef bint to_screen


    cpdef object render(self, Observation obs)
    cpdef close(self)
    cdef void draw_jobs(self, Observation obs, unsigned int start_x, unsigned int start_y, unsigned int width, unsigned int height)
    cdef void draw_machines(self, Observation obs, unsigned int start_x, unsigned int start_y, unsigned int width, unsigned int height)
    cdef void draw_table(self, int[:, ::1] values,unsigned int x,unsigned int y,unsigned int n_rows,unsigned int n_columns, str text, bint active)
    cdef tuple _status_color(self, int status)
    cdef tuple _job_border_color(self, int status)
    cdef tuple cell_color(self, int value)
    cdef int[:, ::1] diffrent_in(self, int[:, ::1] v1, int[:, ::1] v2)
