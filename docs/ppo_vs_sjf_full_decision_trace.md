# Full decision trace — PPO vs SJF, 16 loss episodes

Idle/skip/failed-allocation runs are compressed into one summary line (`idle/wait (N ticks)`); every successful allocation is shown individually with the job it picked.

## Episode 77 (seed 119) — SJF=-185, PPO=-189, gap=-4

Job list (idx: arrival, size): 0:(3,8), 1:(3,10), 2:(0,10), 3:(2,10), 4:(2,5), 5:(3,2), 6:(0,9), 7:(5,1), 8:(8,6), 9:(3,10)

**SJF trace:**
  tick   0: ALLOCATE job 6 (size=9)  reward=+1.00  cumulative=1.00
  tick   1-  9: idle/wait (9 ticks)  reward=-68.00  cumulative=-67.00
  tick  10: ALLOCATE job 7 (size=1)  reward=+1.00  cumulative=-66.00
  tick  11- 11: idle/wait (1 ticks)  reward=-8.00  cumulative=-74.00
  tick  12: ALLOCATE job 5 (size=2)  reward=+1.00  cumulative=-73.00
  tick  13- 14: idle/wait (2 ticks)  reward=-15.00  cumulative=-88.00
  tick  15: ALLOCATE job 4 (size=5)  reward=+1.00  cumulative=-87.00
  tick  16- 20: idle/wait (5 ticks)  reward=-34.00  cumulative=-121.00
  tick  21: ALLOCATE job 8 (size=6)  reward=+1.00  cumulative=-120.00
  tick  22- 27: idle/wait (6 ticks)  reward=-35.00  cumulative=-155.00
  tick  28: ALLOCATE job 0 (size=8)  reward=+1.00  cumulative=-154.00
  tick  29- 36: idle/wait (8 ticks)  reward=-39.00  cumulative=-193.00
  tick  37: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-192.00
  tick  38- 47: idle/wait (10 ticks)  reward=-39.00  cumulative=-231.00
  tick  48: ALLOCATE job 2 (size=10)  reward=+1.00  cumulative=-230.00
  tick  49- 58: idle/wait (10 ticks)  reward=-29.00  cumulative=-259.00
  tick  59: ALLOCATE job 3 (size=10)  reward=+1.00  cumulative=-258.00
  tick  60- 69: idle/wait (10 ticks)  reward=-19.00  cumulative=-277.00
  tick  70: ALLOCATE job 9 (size=10)  reward=+1.00  cumulative=-276.00
  tick  71- 80: idle/wait (10 ticks)  reward=+91.00  cumulative=-185.00

**PPO trace:**
  tick   0-  1: idle/wait (2 ticks)  reward=-6.00  cumulative=-6.00
  tick   2: ALLOCATE job 4 (size=5)  reward=+1.00  cumulative=-5.00
  tick   3-  7: idle/wait (5 ticks)  reward=-42.00  cumulative=-47.00
  tick   8: ALLOCATE job 7 (size=1)  reward=+1.00  cumulative=-46.00
  tick   9-  9: idle/wait (1 ticks)  reward=-8.00  cumulative=-54.00
  tick  10: ALLOCATE job 5 (size=2)  reward=+1.00  cumulative=-53.00
  tick  11- 12: idle/wait (2 ticks)  reward=-15.00  cumulative=-68.00
  tick  13: ALLOCATE job 8 (size=6)  reward=+1.00  cumulative=-67.00
  tick  14- 19: idle/wait (6 ticks)  reward=-41.00  cumulative=-108.00
  tick  20: ALLOCATE job 0 (size=8)  reward=+1.00  cumulative=-107.00
  tick  21- 28: idle/wait (8 ticks)  reward=-47.00  cumulative=-154.00
  tick  29: ALLOCATE job 6 (size=9)  reward=+1.00  cumulative=-153.00
  tick  30- 38: idle/wait (9 ticks)  reward=-44.00  cumulative=-197.00
  tick  39: ALLOCATE job 2 (size=10)  reward=+1.00  cumulative=-196.00
  tick  40- 49: idle/wait (10 ticks)  reward=-39.00  cumulative=-235.00
  tick  50: ALLOCATE job 3 (size=10)  reward=+1.00  cumulative=-234.00
  tick  51- 60: idle/wait (10 ticks)  reward=-29.00  cumulative=-263.00
  tick  61: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-262.00
  tick  62- 71: idle/wait (10 ticks)  reward=-19.00  cumulative=-281.00
  tick  72: ALLOCATE job 9 (size=10)  reward=+1.00  cumulative=-280.00
  tick  73- 82: idle/wait (10 ticks)  reward=+91.00  cumulative=-189.00

---

## Episode 797 (seed 839) — SJF=-164, PPO=-179, gap=-15

Job list (idx: arrival, size): 0:(0,10), 1:(0,10), 2:(0,1), 3:(5,9), 4:(2,10), 5:(6,1), 6:(4,1), 7:(2,10), 8:(8,8), 9:(4,10)

**SJF trace:**
  tick   0: ALLOCATE job 2 (size=1)  reward=+1.00  cumulative=1.00
  tick   1-  1: idle/wait (1 ticks)  reward=-2.00  cumulative=-1.00
  tick   2: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=0.00
  tick   3- 12: idle/wait (10 ticks)  reward=-72.00  cumulative=-72.00
  tick  13: ALLOCATE job 5 (size=1)  reward=+1.00  cumulative=-71.00
  tick  14- 14: idle/wait (1 ticks)  reward=-7.00  cumulative=-78.00
  tick  15: ALLOCATE job 6 (size=1)  reward=+1.00  cumulative=-77.00
  tick  16- 16: idle/wait (1 ticks)  reward=-6.00  cumulative=-83.00
  tick  17: ALLOCATE job 8 (size=8)  reward=+1.00  cumulative=-82.00
  tick  18- 25: idle/wait (8 ticks)  reward=-47.00  cumulative=-129.00
  tick  26: ALLOCATE job 3 (size=9)  reward=+1.00  cumulative=-128.00
  tick  27- 35: idle/wait (9 ticks)  reward=-44.00  cumulative=-172.00
  tick  36: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-171.00
  tick  37- 46: idle/wait (10 ticks)  reward=-39.00  cumulative=-210.00
  tick  47: ALLOCATE job 4 (size=10)  reward=+1.00  cumulative=-209.00
  tick  48- 57: idle/wait (10 ticks)  reward=-29.00  cumulative=-238.00
  tick  58: ALLOCATE job 7 (size=10)  reward=+1.00  cumulative=-237.00
  tick  59- 68: idle/wait (10 ticks)  reward=-19.00  cumulative=-256.00
  tick  69: ALLOCATE job 9 (size=10)  reward=+1.00  cumulative=-255.00
  tick  70- 79: idle/wait (10 ticks)  reward=+91.00  cumulative=-164.00

**PPO trace:**
  tick   0: ALLOCATE job 2 (size=1)  reward=+1.00  cumulative=1.00
  tick   1-  4: idle/wait (4 ticks)  reward=-16.00  cumulative=-15.00
  tick   5: ALLOCATE job 6 (size=1)  reward=+1.00  cumulative=-14.00
  tick   6-  7: idle/wait (2 ticks)  reward=-13.00  cumulative=-27.00
  tick   8: ALLOCATE job 5 (size=1)  reward=+1.00  cumulative=-26.00
  tick   9-  9: idle/wait (1 ticks)  reward=-6.00  cumulative=-32.00
  tick  10: ALLOCATE job 3 (size=9)  reward=+1.00  cumulative=-31.00
  tick  11- 19: idle/wait (9 ticks)  reward=-62.00  cumulative=-93.00
  tick  20: ALLOCATE job 8 (size=8)  reward=+1.00  cumulative=-92.00
  tick  21- 28: idle/wait (8 ticks)  reward=-47.00  cumulative=-139.00
  tick  29: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-138.00
  tick  30- 39: idle/wait (10 ticks)  reward=-49.00  cumulative=-187.00
  tick  40: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-186.00
  tick  41- 50: idle/wait (10 ticks)  reward=-39.00  cumulative=-225.00
  tick  51: ALLOCATE job 4 (size=10)  reward=+1.00  cumulative=-224.00
  tick  52- 61: idle/wait (10 ticks)  reward=-29.00  cumulative=-253.00
  tick  62: ALLOCATE job 7 (size=10)  reward=+1.00  cumulative=-252.00
  tick  63- 72: idle/wait (10 ticks)  reward=-19.00  cumulative=-271.00
  tick  73: ALLOCATE job 9 (size=10)  reward=+1.00  cumulative=-270.00
  tick  74- 83: idle/wait (10 ticks)  reward=+91.00  cumulative=-179.00

---

## Episode 1053 (seed 1095) — SJF=-178, PPO=-181, gap=-3

Job list (idx: arrival, size): 0:(0,10), 1:(0,10), 2:(4,4), 3:(3,10), 4:(4,9), 5:(3,5), 6:(3,1), 7:(0,10), 8:(6,1), 9:(6,10)

**SJF trace:**
  tick   0: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=1.00
  tick   1- 10: idle/wait (10 ticks)  reward=-77.00  cumulative=-76.00
  tick  11: ALLOCATE job 6 (size=1)  reward=+1.00  cumulative=-75.00
  tick  12- 12: idle/wait (1 ticks)  reward=-8.00  cumulative=-83.00
  tick  13: ALLOCATE job 8 (size=1)  reward=+1.00  cumulative=-82.00
  tick  14- 14: idle/wait (1 ticks)  reward=-7.00  cumulative=-89.00
  tick  15: ALLOCATE job 2 (size=4)  reward=+1.00  cumulative=-88.00
  tick  16- 19: idle/wait (4 ticks)  reward=-27.00  cumulative=-115.00
  tick  20: ALLOCATE job 5 (size=5)  reward=+1.00  cumulative=-114.00
  tick  21- 25: idle/wait (5 ticks)  reward=-29.00  cumulative=-143.00
  tick  26: ALLOCATE job 4 (size=9)  reward=+1.00  cumulative=-142.00
  tick  27- 35: idle/wait (9 ticks)  reward=-44.00  cumulative=-186.00
  tick  36: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-185.00
  tick  37- 46: idle/wait (10 ticks)  reward=-39.00  cumulative=-224.00
  tick  47: ALLOCATE job 3 (size=10)  reward=+1.00  cumulative=-223.00
  tick  48- 57: idle/wait (10 ticks)  reward=-29.00  cumulative=-252.00
  tick  58: ALLOCATE job 7 (size=10)  reward=+1.00  cumulative=-251.00
  tick  59- 68: idle/wait (10 ticks)  reward=-19.00  cumulative=-270.00
  tick  69: ALLOCATE job 9 (size=10)  reward=+1.00  cumulative=-269.00
  tick  70- 79: idle/wait (10 ticks)  reward=+91.00  cumulative=-178.00

**PPO trace:**
  tick   0-  2: idle/wait (3 ticks)  reward=-12.00  cumulative=-12.00
  tick   3: ALLOCATE job 6 (size=1)  reward=+1.00  cumulative=-11.00
  tick   4-  4: idle/wait (1 ticks)  reward=-7.00  cumulative=-18.00
  tick   5: ALLOCATE job 2 (size=4)  reward=+1.00  cumulative=-17.00
  tick   6-  9: idle/wait (4 ticks)  reward=-33.00  cumulative=-50.00
  tick  10: ALLOCATE job 8 (size=1)  reward=+1.00  cumulative=-49.00
  tick  11- 11: idle/wait (1 ticks)  reward=-7.00  cumulative=-56.00
  tick  12: ALLOCATE job 5 (size=5)  reward=+1.00  cumulative=-55.00
  tick  13- 17: idle/wait (5 ticks)  reward=-34.00  cumulative=-89.00
  tick  18: ALLOCATE job 4 (size=9)  reward=+1.00  cumulative=-88.00
  tick  19- 27: idle/wait (9 ticks)  reward=-53.00  cumulative=-141.00
  tick  28: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-140.00
  tick  29- 38: idle/wait (10 ticks)  reward=-49.00  cumulative=-189.00
  tick  39: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-188.00
  tick  40- 49: idle/wait (10 ticks)  reward=-39.00  cumulative=-227.00
  tick  50: ALLOCATE job 7 (size=10)  reward=+1.00  cumulative=-226.00
  tick  51- 60: idle/wait (10 ticks)  reward=-29.00  cumulative=-255.00
  tick  61: ALLOCATE job 3 (size=10)  reward=+1.00  cumulative=-254.00
  tick  62- 71: idle/wait (10 ticks)  reward=-19.00  cumulative=-273.00
  tick  72: ALLOCATE job 9 (size=10)  reward=+1.00  cumulative=-272.00
  tick  73- 82: idle/wait (10 ticks)  reward=+91.00  cumulative=-181.00

---

## Episode 1133 (seed 1175) — SJF=-102, PPO=-110, gap=-8

Job list (idx: arrival, size): 0:(2,10), 1:(2,8), 2:(1,1), 3:(0,8), 4:(0,3), 5:(4,8), 6:(4,8), 7:(4,9), 8:(8,1), 9:(7,2)

**SJF trace:**
  tick   0: ALLOCATE job 4 (size=3)  reward=+1.00  cumulative=1.00
  tick   1-  3: idle/wait (3 ticks)  reward=-12.00  cumulative=-11.00
  tick   4: ALLOCATE job 2 (size=1)  reward=+1.00  cumulative=-10.00
  tick   5-  5: idle/wait (1 ticks)  reward=-6.00  cumulative=-16.00
  tick   6: ALLOCATE job 1 (size=8)  reward=+1.00  cumulative=-15.00
  tick   7- 14: idle/wait (8 ticks)  reward=-58.00  cumulative=-73.00
  tick  15: ALLOCATE job 8 (size=1)  reward=+1.00  cumulative=-72.00
  tick  16- 16: idle/wait (1 ticks)  reward=-6.00  cumulative=-78.00
  tick  17: ALLOCATE job 9 (size=2)  reward=+1.00  cumulative=-77.00
  tick  18- 19: idle/wait (2 ticks)  reward=-11.00  cumulative=-88.00
  tick  20: ALLOCATE job 3 (size=8)  reward=+1.00  cumulative=-87.00
  tick  21- 28: idle/wait (8 ticks)  reward=-39.00  cumulative=-126.00
  tick  29: ALLOCATE job 5 (size=8)  reward=+1.00  cumulative=-125.00
  tick  30- 37: idle/wait (8 ticks)  reward=-31.00  cumulative=-156.00
  tick  38: ALLOCATE job 6 (size=8)  reward=+1.00  cumulative=-155.00
  tick  39- 46: idle/wait (8 ticks)  reward=-23.00  cumulative=-178.00
  tick  47: ALLOCATE job 7 (size=9)  reward=+1.00  cumulative=-177.00
  tick  48- 56: idle/wait (9 ticks)  reward=-17.00  cumulative=-194.00
  tick  57: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-193.00
  tick  58- 67: idle/wait (10 ticks)  reward=+91.00  cumulative=-102.00

**PPO trace:**
  tick   0: ALLOCATE job 4 (size=3)  reward=+1.00  cumulative=1.00
  tick   1-  3: idle/wait (3 ticks)  reward=-12.00  cumulative=-11.00
  tick   4: ALLOCATE job 2 (size=1)  reward=+1.00  cumulative=-10.00
  tick   5-  6: idle/wait (2 ticks)  reward=-12.00  cumulative=-22.00
  tick   7: ALLOCATE job 3 (size=8)  reward=+1.00  cumulative=-21.00
  tick   8- 15: idle/wait (8 ticks)  reward=-60.00  cumulative=-81.00
  tick  16: ALLOCATE job 8 (size=1)  reward=+1.00  cumulative=-80.00
  tick  17- 17: idle/wait (1 ticks)  reward=-6.00  cumulative=-86.00
  tick  18: ALLOCATE job 9 (size=2)  reward=+1.00  cumulative=-85.00
  tick  19- 20: idle/wait (2 ticks)  reward=-11.00  cumulative=-96.00
  tick  21: ALLOCATE job 1 (size=8)  reward=+1.00  cumulative=-95.00
  tick  22- 29: idle/wait (8 ticks)  reward=-39.00  cumulative=-134.00
  tick  30: ALLOCATE job 5 (size=8)  reward=+1.00  cumulative=-133.00
  tick  31- 38: idle/wait (8 ticks)  reward=-31.00  cumulative=-164.00
  tick  39: ALLOCATE job 6 (size=8)  reward=+1.00  cumulative=-163.00
  tick  40- 47: idle/wait (8 ticks)  reward=-23.00  cumulative=-186.00
  tick  48: ALLOCATE job 7 (size=9)  reward=+1.00  cumulative=-185.00
  tick  49- 57: idle/wait (9 ticks)  reward=-17.00  cumulative=-202.00
  tick  58: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-201.00
  tick  59- 68: idle/wait (10 ticks)  reward=+91.00  cumulative=-110.00

---

## Episode 1213 (seed 1255) — SJF=-169, PPO=-173, gap=-4

Job list (idx: arrival, size): 0:(0,10), 1:(0,8), 2:(1,7), 3:(0,9), 4:(2,5), 5:(4,1), 6:(5,6), 7:(5,4), 8:(5,6), 9:(3,10)

**SJF trace:**
  tick   0: ALLOCATE job 1 (size=8)  reward=+1.00  cumulative=1.00
  tick   1-  8: idle/wait (8 ticks)  reward=-61.00  cumulative=-60.00
  tick   9: ALLOCATE job 5 (size=1)  reward=+1.00  cumulative=-59.00
  tick  10- 10: idle/wait (1 ticks)  reward=-8.00  cumulative=-67.00
  tick  11: ALLOCATE job 7 (size=4)  reward=+1.00  cumulative=-66.00
  tick  12- 15: idle/wait (4 ticks)  reward=-31.00  cumulative=-97.00
  tick  16: ALLOCATE job 4 (size=5)  reward=+1.00  cumulative=-96.00
  tick  17- 21: idle/wait (5 ticks)  reward=-34.00  cumulative=-130.00
  tick  22: ALLOCATE job 6 (size=6)  reward=+1.00  cumulative=-129.00
  tick  23- 28: idle/wait (6 ticks)  reward=-35.00  cumulative=-164.00
  tick  29: ALLOCATE job 8 (size=6)  reward=+1.00  cumulative=-163.00
  tick  30- 35: idle/wait (6 ticks)  reward=-29.00  cumulative=-192.00
  tick  36: ALLOCATE job 2 (size=7)  reward=+1.00  cumulative=-191.00
  tick  37- 43: idle/wait (7 ticks)  reward=-27.00  cumulative=-218.00
  tick  44: ALLOCATE job 3 (size=9)  reward=+1.00  cumulative=-217.00
  tick  45- 53: idle/wait (9 ticks)  reward=-26.00  cumulative=-243.00
  tick  54: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-242.00
  tick  55- 64: idle/wait (10 ticks)  reward=-19.00  cumulative=-261.00
  tick  65: ALLOCATE job 9 (size=10)  reward=+1.00  cumulative=-260.00
  tick  66- 75: idle/wait (10 ticks)  reward=+91.00  cumulative=-169.00

**PPO trace:**
  tick   0-  0: idle/wait (1 ticks)  reward=-4.00  cumulative=-4.00
  tick   1: ALLOCATE job 2 (size=7)  reward=+1.00  cumulative=-3.00
  tick   2-  8: idle/wait (7 ticks)  reward=-57.00  cumulative=-60.00
  tick   9: ALLOCATE job 5 (size=1)  reward=+1.00  cumulative=-59.00
  tick  10- 10: idle/wait (1 ticks)  reward=-8.00  cumulative=-67.00
  tick  11: ALLOCATE job 7 (size=4)  reward=+1.00  cumulative=-66.00
  tick  12- 15: idle/wait (4 ticks)  reward=-31.00  cumulative=-97.00
  tick  16: ALLOCATE job 4 (size=5)  reward=+1.00  cumulative=-96.00
  tick  17- 21: idle/wait (5 ticks)  reward=-34.00  cumulative=-130.00
  tick  22: ALLOCATE job 6 (size=6)  reward=+1.00  cumulative=-129.00
  tick  23- 28: idle/wait (6 ticks)  reward=-35.00  cumulative=-164.00
  tick  29: ALLOCATE job 8 (size=6)  reward=+1.00  cumulative=-163.00
  tick  30- 35: idle/wait (6 ticks)  reward=-29.00  cumulative=-192.00
  tick  36: ALLOCATE job 1 (size=8)  reward=+1.00  cumulative=-191.00
  tick  37- 44: idle/wait (8 ticks)  reward=-31.00  cumulative=-222.00
  tick  45: ALLOCATE job 3 (size=9)  reward=+1.00  cumulative=-221.00
  tick  46- 54: idle/wait (9 ticks)  reward=-26.00  cumulative=-247.00
  tick  55: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-246.00
  tick  56- 65: idle/wait (10 ticks)  reward=-19.00  cumulative=-265.00
  tick  66: ALLOCATE job 9 (size=10)  reward=+1.00  cumulative=-264.00
  tick  67- 76: idle/wait (10 ticks)  reward=+91.00  cumulative=-173.00

---

## Episode 1525 (seed 1567) — SJF=-86, PPO=-96, gap=-10

Job list (idx: arrival, size): 0:(1,10), 1:(0,9), 2:(1,3), 3:(0,10), 4:(0,7), 5:(5,3), 6:(5,2), 7:(5,1), 8:(4,7), 9:(7,2)

**SJF trace:**
  tick   0: ALLOCATE job 4 (size=7)  reward=+1.00  cumulative=1.00
  tick   1-  7: idle/wait (7 ticks)  reward=-48.00  cumulative=-47.00
  tick   8: ALLOCATE job 7 (size=1)  reward=+1.00  cumulative=-46.00
  tick   9-  9: idle/wait (1 ticks)  reward=-8.00  cumulative=-54.00
  tick  10: ALLOCATE job 6 (size=2)  reward=+1.00  cumulative=-53.00
  tick  11- 12: idle/wait (2 ticks)  reward=-15.00  cumulative=-68.00
  tick  13: ALLOCATE job 9 (size=2)  reward=+1.00  cumulative=-67.00
  tick  14- 15: idle/wait (2 ticks)  reward=-13.00  cumulative=-80.00
  tick  16: ALLOCATE job 2 (size=3)  reward=+1.00  cumulative=-79.00
  tick  17- 19: idle/wait (3 ticks)  reward=-17.00  cumulative=-96.00
  tick  20: ALLOCATE job 5 (size=3)  reward=+1.00  cumulative=-95.00
  tick  21- 23: idle/wait (3 ticks)  reward=-14.00  cumulative=-109.00
  tick  24: ALLOCATE job 8 (size=7)  reward=+1.00  cumulative=-108.00
  tick  25- 31: idle/wait (7 ticks)  reward=-27.00  cumulative=-135.00
  tick  32: ALLOCATE job 1 (size=9)  reward=+1.00  cumulative=-134.00
  tick  33- 41: idle/wait (9 ticks)  reward=-26.00  cumulative=-160.00
  tick  42: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-159.00
  tick  43- 52: idle/wait (10 ticks)  reward=-19.00  cumulative=-178.00
  tick  53: ALLOCATE job 3 (size=10)  reward=+1.00  cumulative=-177.00
  tick  54- 63: idle/wait (10 ticks)  reward=+91.00  cumulative=-86.00

**PPO trace:**
  tick   0-  0: idle/wait (1 ticks)  reward=-5.00  cumulative=-5.00
  tick   1: ALLOCATE job 2 (size=3)  reward=+1.00  cumulative=-4.00
  tick   2-  4: idle/wait (3 ticks)  reward=-15.00  cumulative=-19.00
  tick   5: ALLOCATE job 4 (size=7)  reward=+1.00  cumulative=-18.00
  tick   6- 12: idle/wait (7 ticks)  reward=-60.00  cumulative=-78.00
  tick  13: ALLOCATE job 7 (size=1)  reward=+1.00  cumulative=-77.00
  tick  14- 14: idle/wait (1 ticks)  reward=-7.00  cumulative=-84.00
  tick  15: ALLOCATE job 6 (size=2)  reward=+1.00  cumulative=-83.00
  tick  16- 17: idle/wait (2 ticks)  reward=-13.00  cumulative=-96.00
  tick  18: ALLOCATE job 9 (size=2)  reward=+1.00  cumulative=-95.00
  tick  19- 20: idle/wait (2 ticks)  reward=-11.00  cumulative=-106.00
  tick  21: ALLOCATE job 5 (size=3)  reward=+1.00  cumulative=-105.00
  tick  22- 24: idle/wait (3 ticks)  reward=-14.00  cumulative=-119.00
  tick  25: ALLOCATE job 8 (size=7)  reward=+1.00  cumulative=-118.00
  tick  26- 32: idle/wait (7 ticks)  reward=-27.00  cumulative=-145.00
  tick  33: ALLOCATE job 1 (size=9)  reward=+1.00  cumulative=-144.00
  tick  34- 42: idle/wait (9 ticks)  reward=-26.00  cumulative=-170.00
  tick  43: ALLOCATE job 3 (size=10)  reward=+1.00  cumulative=-169.00
  tick  44- 53: idle/wait (10 ticks)  reward=-19.00  cumulative=-188.00
  tick  54: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-187.00
  tick  55- 64: idle/wait (10 ticks)  reward=+91.00  cumulative=-96.00

---

## Episode 1529 (seed 1571) — SJF=-44, PPO=-56, gap=-12

Job list (idx: arrival, size): 0:(0,10), 1:(0,10), 2:(2,10), 3:(1,2), 4:(0,2), 5:(3,2), 6:(1,2), 7:(5,10), 8:(2,1), 9:(2,1)

**SJF trace:**
  tick   0: ALLOCATE job 4 (size=2)  reward=+1.00  cumulative=1.00
  tick   1-  2: idle/wait (2 ticks)  reward=-12.00  cumulative=-11.00
  tick   3: ALLOCATE job 8 (size=1)  reward=+1.00  cumulative=-10.00
  tick   4-  4: idle/wait (1 ticks)  reward=-7.00  cumulative=-17.00
  tick   5: ALLOCATE job 9 (size=1)  reward=+1.00  cumulative=-16.00
  tick   6-  6: idle/wait (1 ticks)  reward=-6.00  cumulative=-22.00
  tick   7: ALLOCATE job 3 (size=2)  reward=+1.00  cumulative=-21.00
  tick   8-  9: idle/wait (2 ticks)  reward=-13.00  cumulative=-34.00
  tick  10: ALLOCATE job 5 (size=2)  reward=+1.00  cumulative=-33.00
  tick  11- 12: idle/wait (2 ticks)  reward=-11.00  cumulative=-44.00
  tick  13: ALLOCATE job 6 (size=2)  reward=+1.00  cumulative=-43.00
  tick  14- 15: idle/wait (2 ticks)  reward=-9.00  cumulative=-52.00
  tick  16: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-51.00
  tick  17- 26: idle/wait (10 ticks)  reward=-39.00  cumulative=-90.00
  tick  27: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-89.00
  tick  28- 37: idle/wait (10 ticks)  reward=-29.00  cumulative=-118.00
  tick  38: ALLOCATE job 2 (size=10)  reward=+1.00  cumulative=-117.00
  tick  39- 48: idle/wait (10 ticks)  reward=-19.00  cumulative=-136.00
  tick  49: ALLOCATE job 7 (size=10)  reward=+1.00  cumulative=-135.00
  tick  50- 59: idle/wait (10 ticks)  reward=+91.00  cumulative=-44.00

**PPO trace:**
  tick   0: ALLOCATE job 4 (size=2)  reward=+1.00  cumulative=1.00
  tick   1-  2: idle/wait (2 ticks)  reward=-12.00  cumulative=-11.00
  tick   3: ALLOCATE job 8 (size=1)  reward=+1.00  cumulative=-10.00
  tick   4-  4: idle/wait (1 ticks)  reward=-7.00  cumulative=-17.00
  tick   5: ALLOCATE job 9 (size=1)  reward=+1.00  cumulative=-16.00
  tick   6-  6: idle/wait (1 ticks)  reward=-6.00  cumulative=-22.00
  tick   7: ALLOCATE job 3 (size=2)  reward=+1.00  cumulative=-21.00
  tick   8-  9: idle/wait (2 ticks)  reward=-13.00  cumulative=-34.00
  tick  10: ALLOCATE job 6 (size=2)  reward=+1.00  cumulative=-33.00
  tick  11- 12: idle/wait (2 ticks)  reward=-11.00  cumulative=-44.00
  tick  13: ALLOCATE job 5 (size=2)  reward=+1.00  cumulative=-43.00
  tick  14- 18: idle/wait (5 ticks)  reward=-21.00  cumulative=-64.00
  tick  19: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-63.00
  tick  20- 29: idle/wait (10 ticks)  reward=-39.00  cumulative=-102.00
  tick  30: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-101.00
  tick  31- 40: idle/wait (10 ticks)  reward=-29.00  cumulative=-130.00
  tick  41: ALLOCATE job 2 (size=10)  reward=+1.00  cumulative=-129.00
  tick  42- 51: idle/wait (10 ticks)  reward=-19.00  cumulative=-148.00
  tick  52: ALLOCATE job 7 (size=10)  reward=+1.00  cumulative=-147.00
  tick  53- 62: idle/wait (10 ticks)  reward=+91.00  cumulative=-56.00

---

## Episode 1886 (seed 1928) — SJF=12, PPO=6, gap=-6

Job list (idx: arrival, size): 0:(2,10), 1:(0,1), 2:(4,10), 3:(2,4), 4:(0,1), 5:(3,1), 6:(3,3), 7:(1,1), 8:(2,4), 9:(2,2)

**SJF trace:**
  tick   0: ALLOCATE job 1 (size=1)  reward=+1.00  cumulative=1.00
  tick   1-  1: idle/wait (1 ticks)  reward=-2.00  cumulative=-1.00
  tick   2: ALLOCATE job 4 (size=1)  reward=+1.00  cumulative=0.00
  tick   3-  3: idle/wait (1 ticks)  reward=-5.00  cumulative=-5.00
  tick   4: ALLOCATE job 7 (size=1)  reward=+1.00  cumulative=-4.00
  tick   5-  5: idle/wait (1 ticks)  reward=-6.00  cumulative=-10.00
  tick   6: ALLOCATE job 5 (size=1)  reward=+1.00  cumulative=-9.00
  tick   7-  7: idle/wait (1 ticks)  reward=-6.00  cumulative=-15.00
  tick   8: ALLOCATE job 9 (size=2)  reward=+1.00  cumulative=-14.00
  tick   9- 10: idle/wait (2 ticks)  reward=-11.00  cumulative=-25.00
  tick  11: ALLOCATE job 6 (size=3)  reward=+1.00  cumulative=-24.00
  tick  12- 14: idle/wait (3 ticks)  reward=-14.00  cumulative=-38.00
  tick  15: ALLOCATE job 3 (size=4)  reward=+1.00  cumulative=-37.00
  tick  16- 19: idle/wait (4 ticks)  reward=-15.00  cumulative=-52.00
  tick  20: ALLOCATE job 8 (size=4)  reward=+1.00  cumulative=-51.00
  tick  21- 24: idle/wait (4 ticks)  reward=-11.00  cumulative=-62.00
  tick  25: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-61.00
  tick  26- 35: idle/wait (10 ticks)  reward=-19.00  cumulative=-80.00
  tick  36: ALLOCATE job 2 (size=10)  reward=+1.00  cumulative=-79.00
  tick  37- 46: idle/wait (10 ticks)  reward=+91.00  cumulative=12.00

**PPO trace:**
  tick   0: ALLOCATE job 1 (size=1)  reward=+1.00  cumulative=1.00
  tick   1-  1: idle/wait (1 ticks)  reward=-2.00  cumulative=-1.00
  tick   2: ALLOCATE job 4 (size=1)  reward=+1.00  cumulative=0.00
  tick   3-  3: idle/wait (1 ticks)  reward=-5.00  cumulative=-5.00
  tick   4: ALLOCATE job 7 (size=1)  reward=+1.00  cumulative=-4.00
  tick   5-  5: idle/wait (1 ticks)  reward=-6.00  cumulative=-10.00
  tick   6: ALLOCATE job 5 (size=1)  reward=+1.00  cumulative=-9.00
  tick   7-  7: idle/wait (1 ticks)  reward=-6.00  cumulative=-15.00
  tick   8: ALLOCATE job 9 (size=2)  reward=+1.00  cumulative=-14.00
  tick   9- 10: idle/wait (2 ticks)  reward=-11.00  cumulative=-25.00
  tick  11: ALLOCATE job 6 (size=3)  reward=+1.00  cumulative=-24.00
  tick  12- 14: idle/wait (3 ticks)  reward=-14.00  cumulative=-38.00
  tick  15: ALLOCATE job 3 (size=4)  reward=+1.00  cumulative=-37.00
  tick  16- 19: idle/wait (4 ticks)  reward=-15.00  cumulative=-52.00
  tick  20: ALLOCATE job 8 (size=4)  reward=+1.00  cumulative=-51.00
  tick  21- 27: idle/wait (7 ticks)  reward=-17.00  cumulative=-68.00
  tick  28: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-67.00
  tick  29- 38: idle/wait (10 ticks)  reward=-19.00  cumulative=-86.00
  tick  39: ALLOCATE job 2 (size=10)  reward=+1.00  cumulative=-85.00
  tick  40- 49: idle/wait (10 ticks)  reward=+91.00  cumulative=6.00

---

## Episode 2117 (seed 2159) — SJF=-239, PPO=-247, gap=-8

Job list (idx: arrival, size): 0:(0,10), 1:(3,10), 2:(0,9), 3:(2,5), 4:(2,10), 5:(4,4), 6:(1,10), 7:(7,10), 8:(1,10), 9:(5,1)

**SJF trace:**
  tick   0: ALLOCATE job 2 (size=9)  reward=+1.00  cumulative=1.00
  tick   1-  9: idle/wait (9 ticks)  reward=-72.00  cumulative=-71.00
  tick  10: ALLOCATE job 9 (size=1)  reward=+1.00  cumulative=-70.00
  tick  11- 11: idle/wait (1 ticks)  reward=-8.00  cumulative=-78.00
  tick  12: ALLOCATE job 5 (size=4)  reward=+1.00  cumulative=-77.00
  tick  13- 16: idle/wait (4 ticks)  reward=-31.00  cumulative=-108.00
  tick  17: ALLOCATE job 3 (size=5)  reward=+1.00  cumulative=-107.00
  tick  18- 22: idle/wait (5 ticks)  reward=-34.00  cumulative=-141.00
  tick  23: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-140.00
  tick  24- 33: idle/wait (10 ticks)  reward=-59.00  cumulative=-199.00
  tick  34: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-198.00
  tick  35- 44: idle/wait (10 ticks)  reward=-49.00  cumulative=-247.00
  tick  45: ALLOCATE job 4 (size=10)  reward=+1.00  cumulative=-246.00
  tick  46- 55: idle/wait (10 ticks)  reward=-39.00  cumulative=-285.00
  tick  56: ALLOCATE job 6 (size=10)  reward=+1.00  cumulative=-284.00
  tick  57- 66: idle/wait (10 ticks)  reward=-29.00  cumulative=-313.00
  tick  67: ALLOCATE job 7 (size=10)  reward=+1.00  cumulative=-312.00
  tick  68- 77: idle/wait (10 ticks)  reward=-19.00  cumulative=-331.00
  tick  78: ALLOCATE job 8 (size=10)  reward=+1.00  cumulative=-330.00
  tick  79- 88: idle/wait (10 ticks)  reward=+91.00  cumulative=-239.00

**PPO trace:**
  tick   0-  1: idle/wait (2 ticks)  reward=-10.00  cumulative=-10.00
  tick   2: ALLOCATE job 3 (size=5)  reward=+1.00  cumulative=-9.00
  tick   3-  7: idle/wait (5 ticks)  reward=-42.00  cumulative=-51.00
  tick   8: ALLOCATE job 9 (size=1)  reward=+1.00  cumulative=-50.00
  tick   9-  9: idle/wait (1 ticks)  reward=-8.00  cumulative=-58.00
  tick  10: ALLOCATE job 5 (size=4)  reward=+1.00  cumulative=-57.00
  tick  11- 14: idle/wait (4 ticks)  reward=-31.00  cumulative=-88.00
  tick  15: ALLOCATE job 2 (size=9)  reward=+1.00  cumulative=-87.00
  tick  16- 24: idle/wait (9 ticks)  reward=-62.00  cumulative=-149.00
  tick  25: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-148.00
  tick  26- 35: idle/wait (10 ticks)  reward=-59.00  cumulative=-207.00
  tick  36: ALLOCATE job 6 (size=10)  reward=+1.00  cumulative=-206.00
  tick  37- 46: idle/wait (10 ticks)  reward=-49.00  cumulative=-255.00
  tick  47: ALLOCATE job 8 (size=10)  reward=+1.00  cumulative=-254.00
  tick  48- 57: idle/wait (10 ticks)  reward=-39.00  cumulative=-293.00
  tick  58: ALLOCATE job 4 (size=10)  reward=+1.00  cumulative=-292.00
  tick  59- 68: idle/wait (10 ticks)  reward=-29.00  cumulative=-321.00
  tick  69: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-320.00
  tick  70- 79: idle/wait (10 ticks)  reward=-19.00  cumulative=-339.00
  tick  80: ALLOCATE job 7 (size=10)  reward=+1.00  cumulative=-338.00
  tick  81- 90: idle/wait (10 ticks)  reward=+91.00  cumulative=-247.00

---

## Episode 2280 (seed 2322) — SJF=-247, PPO=-261, gap=-14

Job list (idx: arrival, size): 0:(0,10), 1:(0,10), 2:(0,9), 3:(2,10), 4:(1,10), 5:(2,7), 6:(7,10), 7:(5,1), 8:(3,3), 9:(2,10)

**SJF trace:**
  tick   0: ALLOCATE job 2 (size=9)  reward=+1.00  cumulative=1.00
  tick   1-  9: idle/wait (9 ticks)  reward=-74.00  cumulative=-73.00
  tick  10: ALLOCATE job 7 (size=1)  reward=+1.00  cumulative=-72.00
  tick  11- 11: idle/wait (1 ticks)  reward=-8.00  cumulative=-80.00
  tick  12: ALLOCATE job 8 (size=3)  reward=+1.00  cumulative=-79.00
  tick  13- 15: idle/wait (3 ticks)  reward=-23.00  cumulative=-102.00
  tick  16: ALLOCATE job 5 (size=7)  reward=+1.00  cumulative=-101.00
  tick  17- 23: idle/wait (7 ticks)  reward=-48.00  cumulative=-149.00
  tick  24: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-148.00
  tick  25- 34: idle/wait (10 ticks)  reward=-59.00  cumulative=-207.00
  tick  35: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-206.00
  tick  36- 45: idle/wait (10 ticks)  reward=-49.00  cumulative=-255.00
  tick  46: ALLOCATE job 3 (size=10)  reward=+1.00  cumulative=-254.00
  tick  47- 56: idle/wait (10 ticks)  reward=-39.00  cumulative=-293.00
  tick  57: ALLOCATE job 4 (size=10)  reward=+1.00  cumulative=-292.00
  tick  58- 67: idle/wait (10 ticks)  reward=-29.00  cumulative=-321.00
  tick  68: ALLOCATE job 6 (size=10)  reward=+1.00  cumulative=-320.00
  tick  69- 78: idle/wait (10 ticks)  reward=-19.00  cumulative=-339.00
  tick  79: ALLOCATE job 9 (size=10)  reward=+1.00  cumulative=-338.00
  tick  80- 89: idle/wait (10 ticks)  reward=+91.00  cumulative=-247.00

**PPO trace:**
  tick   0-  1: idle/wait (2 ticks)  reward=-11.00  cumulative=-11.00
  tick   2: ALLOCATE job 5 (size=7)  reward=+1.00  cumulative=-10.00
  tick   3-  9: idle/wait (7 ticks)  reward=-63.00  cumulative=-73.00
  tick  10: ALLOCATE job 7 (size=1)  reward=+1.00  cumulative=-72.00
  tick  11- 11: idle/wait (1 ticks)  reward=-8.00  cumulative=-80.00
  tick  12: ALLOCATE job 8 (size=3)  reward=+1.00  cumulative=-79.00
  tick  13- 15: idle/wait (3 ticks)  reward=-23.00  cumulative=-102.00
  tick  16: ALLOCATE job 2 (size=9)  reward=+1.00  cumulative=-101.00
  tick  17- 25: idle/wait (9 ticks)  reward=-62.00  cumulative=-163.00
  tick  26: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-162.00
  tick  27- 36: idle/wait (10 ticks)  reward=-59.00  cumulative=-221.00
  tick  37: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-220.00
  tick  38- 47: idle/wait (10 ticks)  reward=-49.00  cumulative=-269.00
  tick  48: ALLOCATE job 4 (size=10)  reward=+1.00  cumulative=-268.00
  tick  49- 58: idle/wait (10 ticks)  reward=-39.00  cumulative=-307.00
  tick  59: ALLOCATE job 3 (size=10)  reward=+1.00  cumulative=-306.00
  tick  60- 69: idle/wait (10 ticks)  reward=-29.00  cumulative=-335.00
  tick  70: ALLOCATE job 9 (size=10)  reward=+1.00  cumulative=-334.00
  tick  71- 80: idle/wait (10 ticks)  reward=-19.00  cumulative=-353.00
  tick  81: ALLOCATE job 6 (size=10)  reward=+1.00  cumulative=-352.00
  tick  82- 91: idle/wait (10 ticks)  reward=+91.00  cumulative=-261.00

---

## Episode 2646 (seed 2688) — SJF=-86, PPO=-94, gap=-8

Job list (idx: arrival, size): 0:(2,10), 1:(0,10), 2:(4,3), 3:(3,5), 4:(0,7), 5:(4,1), 6:(2,5), 7:(5,1), 8:(0,7), 9:(3,3)

**SJF trace:**
  tick   0: ALLOCATE job 4 (size=7)  reward=+1.00  cumulative=1.00
  tick   1-  7: idle/wait (7 ticks)  reward=-53.00  cumulative=-52.00
  tick   8: ALLOCATE job 5 (size=1)  reward=+1.00  cumulative=-51.00
  tick   9-  9: idle/wait (1 ticks)  reward=-8.00  cumulative=-59.00
  tick  10: ALLOCATE job 7 (size=1)  reward=+1.00  cumulative=-58.00
  tick  11- 11: idle/wait (1 ticks)  reward=-7.00  cumulative=-65.00
  tick  12: ALLOCATE job 2 (size=3)  reward=+1.00  cumulative=-64.00
  tick  13- 15: idle/wait (3 ticks)  reward=-20.00  cumulative=-84.00
  tick  16: ALLOCATE job 9 (size=3)  reward=+1.00  cumulative=-83.00
  tick  17- 19: idle/wait (3 ticks)  reward=-17.00  cumulative=-100.00
  tick  20: ALLOCATE job 3 (size=5)  reward=+1.00  cumulative=-99.00
  tick  21- 25: idle/wait (5 ticks)  reward=-24.00  cumulative=-123.00
  tick  26: ALLOCATE job 6 (size=5)  reward=+1.00  cumulative=-122.00
  tick  27- 31: idle/wait (5 ticks)  reward=-19.00  cumulative=-141.00
  tick  32: ALLOCATE job 8 (size=7)  reward=+1.00  cumulative=-140.00
  tick  33- 39: idle/wait (7 ticks)  reward=-20.00  cumulative=-160.00
  tick  40: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-159.00
  tick  41- 50: idle/wait (10 ticks)  reward=-19.00  cumulative=-178.00
  tick  51: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-177.00
  tick  52- 61: idle/wait (10 ticks)  reward=+91.00  cumulative=-86.00

**PPO trace:**
  tick   0-  1: idle/wait (2 ticks)  reward=-8.00  cumulative=-8.00
  tick   2: ALLOCATE job 6 (size=5)  reward=+1.00  cumulative=-7.00
  tick   3-  7: idle/wait (5 ticks)  reward=-45.00  cumulative=-52.00
  tick   8: ALLOCATE job 5 (size=1)  reward=+1.00  cumulative=-51.00
  tick   9-  9: idle/wait (1 ticks)  reward=-8.00  cumulative=-59.00
  tick  10: ALLOCATE job 7 (size=1)  reward=+1.00  cumulative=-58.00
  tick  11- 11: idle/wait (1 ticks)  reward=-7.00  cumulative=-65.00
  tick  12: ALLOCATE job 9 (size=3)  reward=+1.00  cumulative=-64.00
  tick  13- 15: idle/wait (3 ticks)  reward=-20.00  cumulative=-84.00
  tick  16: ALLOCATE job 2 (size=3)  reward=+1.00  cumulative=-83.00
  tick  17- 19: idle/wait (3 ticks)  reward=-17.00  cumulative=-100.00
  tick  20: ALLOCATE job 3 (size=5)  reward=+1.00  cumulative=-99.00
  tick  21- 25: idle/wait (5 ticks)  reward=-24.00  cumulative=-123.00
  tick  26: ALLOCATE job 4 (size=7)  reward=+1.00  cumulative=-122.00
  tick  27- 33: idle/wait (7 ticks)  reward=-27.00  cumulative=-149.00
  tick  34: ALLOCATE job 8 (size=7)  reward=+1.00  cumulative=-148.00
  tick  35- 41: idle/wait (7 ticks)  reward=-20.00  cumulative=-168.00
  tick  42: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-167.00
  tick  43- 52: idle/wait (10 ticks)  reward=-19.00  cumulative=-186.00
  tick  53: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-185.00
  tick  54- 63: idle/wait (10 ticks)  reward=+91.00  cumulative=-94.00

---

## Episode 3255 (seed 3297) — SJF=-39, PPO=-55, gap=-16

Job list (idx: arrival, size): 0:(0,10), 1:(3,10), 2:(2,10), 3:(0,2), 4:(4,10), 5:(2,2), 6:(3,3), 7:(1,1), 8:(1,1), 9:(5,1)

**SJF trace:**
  tick   0: ALLOCATE job 3 (size=2)  reward=+1.00  cumulative=1.00
  tick   1-  2: idle/wait (2 ticks)  reward=-9.00  cumulative=-8.00
  tick   3: ALLOCATE job 7 (size=1)  reward=+1.00  cumulative=-7.00
  tick   4-  4: idle/wait (1 ticks)  reward=-6.00  cumulative=-13.00
  tick   5: ALLOCATE job 8 (size=1)  reward=+1.00  cumulative=-12.00
  tick   6-  6: idle/wait (1 ticks)  reward=-6.00  cumulative=-18.00
  tick   7: ALLOCATE job 5 (size=2)  reward=+1.00  cumulative=-17.00
  tick   8-  9: idle/wait (2 ticks)  reward=-13.00  cumulative=-30.00
  tick  10: ALLOCATE job 9 (size=1)  reward=+1.00  cumulative=-29.00
  tick  11- 11: idle/wait (1 ticks)  reward=-5.00  cumulative=-34.00
  tick  12: ALLOCATE job 6 (size=3)  reward=+1.00  cumulative=-33.00
  tick  13- 15: idle/wait (3 ticks)  reward=-14.00  cumulative=-47.00
  tick  16: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-46.00
  tick  17- 26: idle/wait (10 ticks)  reward=-39.00  cumulative=-85.00
  tick  27: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-84.00
  tick  28- 37: idle/wait (10 ticks)  reward=-29.00  cumulative=-113.00
  tick  38: ALLOCATE job 2 (size=10)  reward=+1.00  cumulative=-112.00
  tick  39- 48: idle/wait (10 ticks)  reward=-19.00  cumulative=-131.00
  tick  49: ALLOCATE job 4 (size=10)  reward=+1.00  cumulative=-130.00
  tick  50- 59: idle/wait (10 ticks)  reward=+91.00  cumulative=-39.00

**PPO trace:**
  tick   0: ALLOCATE job 3 (size=2)  reward=+1.00  cumulative=1.00
  tick   1-  2: idle/wait (2 ticks)  reward=-9.00  cumulative=-8.00
  tick   3: ALLOCATE job 7 (size=1)  reward=+1.00  cumulative=-7.00
  tick   4-  4: idle/wait (1 ticks)  reward=-6.00  cumulative=-13.00
  tick   5: ALLOCATE job 8 (size=1)  reward=+1.00  cumulative=-12.00
  tick   6-  6: idle/wait (1 ticks)  reward=-6.00  cumulative=-18.00
  tick   7: ALLOCATE job 5 (size=2)  reward=+1.00  cumulative=-17.00
  tick   8-  9: idle/wait (2 ticks)  reward=-13.00  cumulative=-30.00
  tick  10: ALLOCATE job 9 (size=1)  reward=+1.00  cumulative=-29.00
  tick  11- 11: idle/wait (1 ticks)  reward=-5.00  cumulative=-34.00
  tick  12: ALLOCATE job 6 (size=3)  reward=+1.00  cumulative=-33.00
  tick  13- 19: idle/wait (7 ticks)  reward=-30.00  cumulative=-63.00
  tick  20: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-62.00
  tick  21- 30: idle/wait (10 ticks)  reward=-39.00  cumulative=-101.00
  tick  31: ALLOCATE job 2 (size=10)  reward=+1.00  cumulative=-100.00
  tick  32- 41: idle/wait (10 ticks)  reward=-29.00  cumulative=-129.00
  tick  42: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-128.00
  tick  43- 52: idle/wait (10 ticks)  reward=-19.00  cumulative=-147.00
  tick  53: ALLOCATE job 4 (size=10)  reward=+1.00  cumulative=-146.00
  tick  54- 63: idle/wait (10 ticks)  reward=+91.00  cumulative=-55.00

---

## Episode 3259 (seed 3301) — SJF=-269, PPO=-274, gap=-5

Job list (idx: arrival, size): 0:(0,10), 1:(0,10), 2:(3,10), 3:(0,10), 4:(5,7), 5:(6,5), 6:(1,10), 7:(5,4), 8:(2,6), 9:(6,10)

**SJF trace:**
  tick   0: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=1.00
  tick   1- 10: idle/wait (10 ticks)  reward=-78.00  cumulative=-77.00
  tick  11: ALLOCATE job 7 (size=4)  reward=+1.00  cumulative=-76.00
  tick  12- 15: idle/wait (4 ticks)  reward=-35.00  cumulative=-111.00
  tick  16: ALLOCATE job 5 (size=5)  reward=+1.00  cumulative=-110.00
  tick  17- 21: idle/wait (5 ticks)  reward=-39.00  cumulative=-149.00
  tick  22: ALLOCATE job 8 (size=6)  reward=+1.00  cumulative=-148.00
  tick  23- 28: idle/wait (6 ticks)  reward=-41.00  cumulative=-189.00
  tick  29: ALLOCATE job 4 (size=7)  reward=+1.00  cumulative=-188.00
  tick  30- 36: idle/wait (7 ticks)  reward=-41.00  cumulative=-229.00
  tick  37: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-228.00
  tick  38- 47: idle/wait (10 ticks)  reward=-49.00  cumulative=-277.00
  tick  48: ALLOCATE job 2 (size=10)  reward=+1.00  cumulative=-276.00
  tick  49- 58: idle/wait (10 ticks)  reward=-39.00  cumulative=-315.00
  tick  59: ALLOCATE job 3 (size=10)  reward=+1.00  cumulative=-314.00
  tick  60- 69: idle/wait (10 ticks)  reward=-29.00  cumulative=-343.00
  tick  70: ALLOCATE job 6 (size=10)  reward=+1.00  cumulative=-342.00
  tick  71- 80: idle/wait (10 ticks)  reward=-19.00  cumulative=-361.00
  tick  81: ALLOCATE job 9 (size=10)  reward=+1.00  cumulative=-360.00
  tick  82- 91: idle/wait (10 ticks)  reward=+91.00  cumulative=-269.00

**PPO trace:**
  tick   0-  1: idle/wait (2 ticks)  reward=-9.00  cumulative=-9.00
  tick   2: ALLOCATE job 8 (size=6)  reward=+1.00  cumulative=-8.00
  tick   3-  8: idle/wait (6 ticks)  reward=-49.00  cumulative=-57.00
  tick   9: ALLOCATE job 7 (size=4)  reward=+1.00  cumulative=-56.00
  tick  10- 13: idle/wait (4 ticks)  reward=-35.00  cumulative=-91.00
  tick  14: ALLOCATE job 5 (size=5)  reward=+1.00  cumulative=-90.00
  tick  15- 19: idle/wait (5 ticks)  reward=-39.00  cumulative=-129.00
  tick  20: ALLOCATE job 4 (size=7)  reward=+1.00  cumulative=-128.00
  tick  21- 27: idle/wait (7 ticks)  reward=-48.00  cumulative=-176.00
  tick  28: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-175.00
  tick  29- 38: idle/wait (10 ticks)  reward=-59.00  cumulative=-234.00
  tick  39: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-233.00
  tick  40- 49: idle/wait (10 ticks)  reward=-49.00  cumulative=-282.00
  tick  50: ALLOCATE job 3 (size=10)  reward=+1.00  cumulative=-281.00
  tick  51- 60: idle/wait (10 ticks)  reward=-39.00  cumulative=-320.00
  tick  61: ALLOCATE job 6 (size=10)  reward=+1.00  cumulative=-319.00
  tick  62- 71: idle/wait (10 ticks)  reward=-29.00  cumulative=-348.00
  tick  72: ALLOCATE job 2 (size=10)  reward=+1.00  cumulative=-347.00
  tick  73- 82: idle/wait (10 ticks)  reward=-19.00  cumulative=-366.00
  tick  83: ALLOCATE job 9 (size=10)  reward=+1.00  cumulative=-365.00
  tick  84- 93: idle/wait (10 ticks)  reward=+91.00  cumulative=-274.00

---

## Episode 3296 (seed 3338) — SJF=-199, PPO=-204, gap=-5

Job list (idx: arrival, size): 0:(0,10), 1:(0,10), 2:(0,8), 3:(0,10), 4:(5,1), 5:(0,9), 6:(5,7), 7:(3,8), 8:(1,2), 9:(4,7)

**SJF trace:**
  tick   0: ALLOCATE job 2 (size=8)  reward=+1.00  cumulative=1.00
  tick   1-  8: idle/wait (8 ticks)  reward=-66.00  cumulative=-65.00
  tick   9: ALLOCATE job 4 (size=1)  reward=+1.00  cumulative=-64.00
  tick  10- 10: idle/wait (1 ticks)  reward=-8.00  cumulative=-72.00
  tick  11: ALLOCATE job 8 (size=2)  reward=+1.00  cumulative=-71.00
  tick  12- 13: idle/wait (2 ticks)  reward=-15.00  cumulative=-86.00
  tick  14: ALLOCATE job 6 (size=7)  reward=+1.00  cumulative=-85.00
  tick  15- 21: idle/wait (7 ticks)  reward=-48.00  cumulative=-133.00
  tick  22: ALLOCATE job 9 (size=7)  reward=+1.00  cumulative=-132.00
  tick  23- 29: idle/wait (7 ticks)  reward=-41.00  cumulative=-173.00
  tick  30: ALLOCATE job 7 (size=8)  reward=+1.00  cumulative=-172.00
  tick  31- 38: idle/wait (8 ticks)  reward=-39.00  cumulative=-211.00
  tick  39: ALLOCATE job 5 (size=9)  reward=+1.00  cumulative=-210.00
  tick  40- 48: idle/wait (9 ticks)  reward=-35.00  cumulative=-245.00
  tick  49: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-244.00
  tick  50- 59: idle/wait (10 ticks)  reward=-29.00  cumulative=-273.00
  tick  60: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-272.00
  tick  61- 70: idle/wait (10 ticks)  reward=-19.00  cumulative=-291.00
  tick  71: ALLOCATE job 3 (size=10)  reward=+1.00  cumulative=-290.00
  tick  72- 81: idle/wait (10 ticks)  reward=+91.00  cumulative=-199.00

**PPO trace:**
  tick   0-  0: idle/wait (1 ticks)  reward=-6.00  cumulative=-6.00
  tick   1: ALLOCATE job 8 (size=2)  reward=+1.00  cumulative=-5.00
  tick   2-  3: idle/wait (2 ticks)  reward=-12.00  cumulative=-17.00
  tick   4: ALLOCATE job 2 (size=8)  reward=+1.00  cumulative=-16.00
  tick   5- 12: idle/wait (8 ticks)  reward=-69.00  cumulative=-85.00
  tick  13: ALLOCATE job 4 (size=1)  reward=+1.00  cumulative=-84.00
  tick  14- 14: idle/wait (1 ticks)  reward=-7.00  cumulative=-91.00
  tick  15: ALLOCATE job 9 (size=7)  reward=+1.00  cumulative=-90.00
  tick  16- 22: idle/wait (7 ticks)  reward=-48.00  cumulative=-138.00
  tick  23: ALLOCATE job 6 (size=7)  reward=+1.00  cumulative=-137.00
  tick  24- 30: idle/wait (7 ticks)  reward=-41.00  cumulative=-178.00
  tick  31: ALLOCATE job 7 (size=8)  reward=+1.00  cumulative=-177.00
  tick  32- 39: idle/wait (8 ticks)  reward=-39.00  cumulative=-216.00
  tick  40: ALLOCATE job 5 (size=9)  reward=+1.00  cumulative=-215.00
  tick  41- 49: idle/wait (9 ticks)  reward=-35.00  cumulative=-250.00
  tick  50: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-249.00
  tick  51- 60: idle/wait (10 ticks)  reward=-29.00  cumulative=-278.00
  tick  61: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-277.00
  tick  62- 71: idle/wait (10 ticks)  reward=-19.00  cumulative=-296.00
  tick  72: ALLOCATE job 3 (size=10)  reward=+1.00  cumulative=-295.00
  tick  73- 82: idle/wait (10 ticks)  reward=+91.00  cumulative=-204.00

---

## Episode 4276 (seed 4318) — SJF=-212, PPO=-229, gap=-17

Job list (idx: arrival, size): 0:(0,10), 1:(0,10), 2:(1,9), 3:(3,5), 4:(0,10), 5:(5,3), 6:(0,9), 7:(3,9), 8:(3,8), 9:(6,1)

**SJF trace:**
  tick   0: ALLOCATE job 6 (size=9)  reward=+1.00  cumulative=1.00
  tick   1-  9: idle/wait (9 ticks)  reward=-74.00  cumulative=-73.00
  tick  10: ALLOCATE job 9 (size=1)  reward=+1.00  cumulative=-72.00
  tick  11- 11: idle/wait (1 ticks)  reward=-8.00  cumulative=-80.00
  tick  12: ALLOCATE job 5 (size=3)  reward=+1.00  cumulative=-79.00
  tick  13- 15: idle/wait (3 ticks)  reward=-23.00  cumulative=-102.00
  tick  16: ALLOCATE job 3 (size=5)  reward=+1.00  cumulative=-101.00
  tick  17- 21: idle/wait (5 ticks)  reward=-34.00  cumulative=-135.00
  tick  22: ALLOCATE job 8 (size=8)  reward=+1.00  cumulative=-134.00
  tick  23- 30: idle/wait (8 ticks)  reward=-47.00  cumulative=-181.00
  tick  31: ALLOCATE job 2 (size=9)  reward=+1.00  cumulative=-180.00
  tick  32- 40: idle/wait (9 ticks)  reward=-44.00  cumulative=-224.00
  tick  41: ALLOCATE job 7 (size=9)  reward=+1.00  cumulative=-223.00
  tick  42- 50: idle/wait (9 ticks)  reward=-35.00  cumulative=-258.00
  tick  51: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-257.00
  tick  52- 61: idle/wait (10 ticks)  reward=-29.00  cumulative=-286.00
  tick  62: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-285.00
  tick  63- 72: idle/wait (10 ticks)  reward=-19.00  cumulative=-304.00
  tick  73: ALLOCATE job 4 (size=10)  reward=+1.00  cumulative=-303.00
  tick  74- 83: idle/wait (10 ticks)  reward=+91.00  cumulative=-212.00

**PPO trace:**
  tick   0-  2: idle/wait (3 ticks)  reward=-18.00  cumulative=-18.00
  tick   3: ALLOCATE job 3 (size=5)  reward=+1.00  cumulative=-17.00
  tick   4-  8: idle/wait (5 ticks)  reward=-46.00  cumulative=-63.00
  tick   9: ALLOCATE job 9 (size=1)  reward=+1.00  cumulative=-62.00
  tick  10- 10: idle/wait (1 ticks)  reward=-8.00  cumulative=-70.00
  tick  11: ALLOCATE job 5 (size=3)  reward=+1.00  cumulative=-69.00
  tick  12- 14: idle/wait (3 ticks)  reward=-23.00  cumulative=-92.00
  tick  15: ALLOCATE job 8 (size=8)  reward=+1.00  cumulative=-91.00
  tick  16- 23: idle/wait (8 ticks)  reward=-55.00  cumulative=-146.00
  tick  24: ALLOCATE job 6 (size=9)  reward=+1.00  cumulative=-145.00
  tick  25- 33: idle/wait (9 ticks)  reward=-53.00  cumulative=-198.00
  tick  34: ALLOCATE job 2 (size=9)  reward=+1.00  cumulative=-197.00
  tick  35- 43: idle/wait (9 ticks)  reward=-44.00  cumulative=-241.00
  tick  44: ALLOCATE job 7 (size=9)  reward=+1.00  cumulative=-240.00
  tick  45- 53: idle/wait (9 ticks)  reward=-35.00  cumulative=-275.00
  tick  54: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-274.00
  tick  55- 64: idle/wait (10 ticks)  reward=-29.00  cumulative=-303.00
  tick  65: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-302.00
  tick  66- 75: idle/wait (10 ticks)  reward=-19.00  cumulative=-321.00
  tick  76: ALLOCATE job 4 (size=10)  reward=+1.00  cumulative=-320.00
  tick  77- 86: idle/wait (10 ticks)  reward=+91.00  cumulative=-229.00

---

## Episode 4524 (seed 4566) — SJF=-199, PPO=-214, gap=-15

Job list (idx: arrival, size): 0:(0,10), 1:(3,10), 2:(2,10), 3:(0,10), 4:(5,4), 5:(4,3), 6:(5,5), 7:(5,1), 8:(0,10), 9:(5,10)

**SJF trace:**
  tick   0: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=1.00
  tick   1- 10: idle/wait (10 ticks)  reward=-77.00  cumulative=-76.00
  tick  11: ALLOCATE job 7 (size=1)  reward=+1.00  cumulative=-75.00
  tick  12- 12: idle/wait (1 ticks)  reward=-8.00  cumulative=-83.00
  tick  13: ALLOCATE job 5 (size=3)  reward=+1.00  cumulative=-82.00
  tick  14- 16: idle/wait (3 ticks)  reward=-23.00  cumulative=-105.00
  tick  17: ALLOCATE job 4 (size=4)  reward=+1.00  cumulative=-104.00
  tick  18- 21: idle/wait (4 ticks)  reward=-27.00  cumulative=-131.00
  tick  22: ALLOCATE job 6 (size=5)  reward=+1.00  cumulative=-130.00
  tick  23- 27: idle/wait (5 ticks)  reward=-29.00  cumulative=-159.00
  tick  28: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-158.00
  tick  29- 38: idle/wait (10 ticks)  reward=-49.00  cumulative=-207.00
  tick  39: ALLOCATE job 2 (size=10)  reward=+1.00  cumulative=-206.00
  tick  40- 49: idle/wait (10 ticks)  reward=-39.00  cumulative=-245.00
  tick  50: ALLOCATE job 3 (size=10)  reward=+1.00  cumulative=-244.00
  tick  51- 60: idle/wait (10 ticks)  reward=-29.00  cumulative=-273.00
  tick  61: ALLOCATE job 8 (size=10)  reward=+1.00  cumulative=-272.00
  tick  62- 71: idle/wait (10 ticks)  reward=-19.00  cumulative=-291.00
  tick  72: ALLOCATE job 9 (size=10)  reward=+1.00  cumulative=-290.00
  tick  73- 82: idle/wait (10 ticks)  reward=+91.00  cumulative=-199.00

**PPO trace:**
  tick   0-  3: idle/wait (4 ticks)  reward=-18.00  cumulative=-18.00
  tick   4: ALLOCATE job 5 (size=3)  reward=+1.00  cumulative=-17.00
  tick   5-  7: idle/wait (3 ticks)  reward=-29.00  cumulative=-46.00
  tick   8: ALLOCATE job 7 (size=1)  reward=+1.00  cumulative=-45.00
  tick   9-  9: idle/wait (1 ticks)  reward=-8.00  cumulative=-53.00
  tick  10: ALLOCATE job 4 (size=4)  reward=+1.00  cumulative=-52.00
  tick  11- 14: idle/wait (4 ticks)  reward=-31.00  cumulative=-83.00
  tick  15: ALLOCATE job 6 (size=5)  reward=+1.00  cumulative=-82.00
  tick  16- 20: idle/wait (5 ticks)  reward=-34.00  cumulative=-116.00
  tick  21: ALLOCATE job 0 (size=10)  reward=+1.00  cumulative=-115.00
  tick  22- 31: idle/wait (10 ticks)  reward=-59.00  cumulative=-174.00
  tick  32: ALLOCATE job 3 (size=10)  reward=+1.00  cumulative=-173.00
  tick  33- 42: idle/wait (10 ticks)  reward=-49.00  cumulative=-222.00
  tick  43: ALLOCATE job 8 (size=10)  reward=+1.00  cumulative=-221.00
  tick  44- 53: idle/wait (10 ticks)  reward=-39.00  cumulative=-260.00
  tick  54: ALLOCATE job 2 (size=10)  reward=+1.00  cumulative=-259.00
  tick  55- 64: idle/wait (10 ticks)  reward=-29.00  cumulative=-288.00
  tick  65: ALLOCATE job 1 (size=10)  reward=+1.00  cumulative=-287.00
  tick  66- 75: idle/wait (10 ticks)  reward=-19.00  cumulative=-306.00
  tick  76: ALLOCATE job 9 (size=10)  reward=+1.00  cumulative=-305.00
  tick  77- 86: idle/wait (10 ticks)  reward=+91.00  cumulative=-214.00

---

