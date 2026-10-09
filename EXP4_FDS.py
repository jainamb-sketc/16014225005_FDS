import math
import csv
import os
 
os.system("")  # enables ANSI colours in Windows terminals (VS Code)
 
# ---------- colour helpers ----------
RESET, BOLD = "\033[0m", "\033[1m"
RED, GREEN, YELLOW = "\033[91m", "\033[92m", "\033[93m"
BLUE, MAGENTA, CYAN, WHITE = "\033[94m", "\033[95m", "\033[96m", "\033[97m"
 
 
def c(text, *styles):
    return "".join(styles) + str(text) + RESET
 
 
FILE_NAME = "london_weather_data_1979_to_2023.csv"
 
 
def load_columns(file_name, col_x, col_y):
    """Read two columns from the CSV, skipping rows with missing values."""
    x_vals, y_vals = [], []
    with open(file_name, newline="") as f:
        for row in csv.DictReader(f):
            if row[col_x] == "" or row[col_y] == "":
                continue
            x_vals.append(float(row[col_x]))
            y_vals.append(float(row[col_y]))
    return x_vals, y_vals
 
 
# ---------- correlation coefficient (no in-built function) ----------
#            n*Σxy - (Σx)(Σy)
#  r = ---------------------------------------------
#      sqrt[ n*Σx² - (Σx)² ] * sqrt[ n*Σy² - (Σy)² ]
def correlation_coefficient(x, y):
    n = len(x)
    sum_x, sum_y = sum(x), sum(y)
    sum_xy = sum(a * b for a, b in zip(x, y))
    sum_x2 = sum(a * a for a in x)
    sum_y2 = sum(b * b for b in y)
    num = n * sum_xy - sum_x * sum_y
    den = math.sqrt(n * sum_x2 - sum_x ** 2) * math.sqrt(n * sum_y2 - sum_y ** 2)
    return num / den
 
 
def describe(r):
    a = abs(r)
    if a < 0.2:
        return "No (or negligible) linear correlation"
    strength = "Strong" if a >= 0.7 else "Moderate" if a >= 0.4 else "Weak"
    return f"{strength} {'positive' if r > 0 else 'negative'} correlation"
 
 
# ---------- scatter plot drawn in the terminal ----------
def terminal_scatter(x, y, colour, width=60, height=18):
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    grid = [[0] * width for _ in range(height)]
    for a, b in zip(x, y):
        col = int((a - min_x) / (max_x - min_x) * (width - 1))
        row = int((b - min_y) / (max_y - min_y) * (height - 1))
        grid[height - 1 - row][col] += 1
 
    peak = max(max(r) for r in grid)
    shades = " .:oO@"   # sparse -> dense
    for i, row in enumerate(grid):
        label = f"{max_y:>8.0f} " if i == 0 else f"{min_y:>8.0f} " if i == height - 1 else " " * 9
        line = ""
        for cell in row:
            if cell == 0:
                line += " "
            else:
                level = 1 + int((cell / peak) ** 0.5 * (len(shades) - 2))
                line += shades[level]
        print(c(label, WHITE) + c("|", WHITE) + c(line, colour))
    print(" " * 9 + c("+" + "-" * width, WHITE))
    print(" " * 10 + c(f"{min_x:<.0f}", WHITE) + " " * (width - 8) + c(f"{max_x:>.0f}", WHITE))
 
 
cases = [
    ("POSITIVE CORRELATION", "TG", "TX",
     "Mean temperature TG (0.1 °C)", "Max temperature TX (0.1 °C)", GREEN),
    ("NEGATIVE CORRELATION", "SS", "CC",
     "Sunshine duration SS (0.1 hours)", "Cloud cover CC (oktas)", RED),
    ("NO CORRELATION", "TG", "PP",
     "Mean temperature TG (0.1 °C)", "Sea level pressure PP (0.1 hPa)", BLUE),
]
 
print()
print(c("=" * 70, CYAN, BOLD))
print(c("   EXPERIMENT 4 : CORRELATION COEFFICIENT - LONDON WEATHER DATA", CYAN, BOLD))
print(c("=" * 70, CYAN, BOLD))
 
summary = []
for title, cx, cy, lx, ly, colour in cases:
    x, y = load_columns(FILE_NAME, cx, cy)
    r = correlation_coefficient(x, y)
    summary.append((title, cx, cy, r))
 
    print()
    print(c(f"{title}  ({cx} vs {cy})", colour, BOLD))
    print(c("-" * 70, colour))
    print(f"  {c('Data points   :', YELLOW)} {len(x)}")
    print(f"  {c('Correlation r :', YELLOW)} {c(f'{r:.4f}', colour, BOLD)}")
    print(f"  {c('Meaning       :', YELLOW)} {c(describe(r), colour)}")
    print()
    print(c(f"  Scatter plot  (Y: {ly})", MAGENTA))
    terminal_scatter(x, y, colour)
    print(c(f"  X: {lx}", MAGENTA))
 
print()
print(c("=" * 70, CYAN, BOLD))
print(c("   SUMMARY", CYAN, BOLD))
print(c("=" * 70, CYAN, BOLD))
for (title, cx, cy, r), (_, _, _, _, _, colour) in zip(summary, cases):
    print(f"  {c(f'{title:<22}', colour, BOLD)} {cx} vs {cy:<3}  r = {c(f'{r:+.4f}', colour, BOLD)}")
print()
