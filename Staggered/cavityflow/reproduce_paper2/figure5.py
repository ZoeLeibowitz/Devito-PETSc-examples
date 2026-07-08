# repoduce figure 5 from https://pdf.sciencedirectassets.com/272570/1-s2.0-S0021999100X02513/1-s2.0-0021999185901482/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEGsaCXVzLWVhc3QtMSJGMEQCIBS1F7mGptLOMbJ6PunezwOAjZYzhWrZuMY%2B8lzItRNMAiAGFG%2FEVxQoL1h%2BBvzDij%2Bv14K7ycj6ZxBVeOeuxf8izyqyBQgzEAUaDDA1OTAwMzU0Njg2NSIMNMMwndYgdsGt%2FCfzKo8FQACH4ZJmbQo32vbvrQn6sTRdUTCazmZnmy3RUDkBoMP6u5O8EndwsFvmZ3njNpCReNgu%2F7WWGcDcxzdJLS50DstxG7Ul2LjZJxTWZ%2FoTow6yOdlPgbwgCXeMXLWnridULXEg%2BHp%2BL8HnW43%2BRXBIg5sH%2B2Q7p85mIIGtjI8xZxzcZSNQdw2GYJQ%2FxScu%2FH5dv1LEgeqtaJo50CzNUDVAFJiYQyfnn2SqHDs86GwTKaZnNjI3KyDf8GBAHeuHJv1CSgC6y7p3r0QAyQzB28jwp%2BHL6BoTo6Y8XC4NZCRHcLyucetWSPJZ%2BNh56PtXErzjzUiQtOJr%2BLQhNMFXUsAwLpKv%2BgLFQqHhNXgb76hgrv5kGq3IgTvqRldgKSVTAxYF9i6Ac%2Bpc3k8Gdyp31%2BKmYc1qdLRl1gWxujliDlFjLubmva4CfCkGfINYtyqsgsY98Dz8uYR90%2Fe8Ys2%2FsfBQcgb%2FGCP15UQM6uC4Mz3P1UZ89ccfMIDLxo36SBzHXdO%2BaYXTsaxhAGqdB6qN%2BSwPZ81E8AlIhvlYTHRH9xzRS5v7YMZ010Ne0%2BRIysG28LZhvo36sNPtagDh%2BpdotIfo6hh0SsJM07Mn0xYwwRuKZcfnbuJ6kGkBWAlTM9GdIQOs5JchjcjvXRp%2B%2F%2FyhoeEd7tY4lOK7QQ52JPKIu%2FG4IPoFlHXMvDcIWiY4VPwG9KJpwOXT4hvmxBma3FSfeNow9dvAYo%2Bbt9LgDZM9QQQh9SWFuRlYfzdr5NazVeyHj5Z1yN0XofTWwjEoP7GMZrvZkTn%2F3MKE5TibD5V4Q%2BRtVCDOjtr1%2BL0FUHLD6t6aWLoYdaFxQFXulCiNX9sPUX5ePWMju6BpxHkjzZT3gqgtYjCAvfTNBjqyAaW1XqS2gtHvAzgbOHZ%2B07ql73%2B0k7UtAfA8ZIeO1EgGwVkcSw3nmgnc3sjw3qg8dRWE%2F5KmxEN2Yjx4j5qO4ZJB7VbugAkOlkzPhW%2B5WRSGOWg6nr3WtctxBYtAOKUBeOUmRg069ncsOIlk%2BfwqOKzg%2Bc9%2BpUyV6%2F0f3ZVQlPb7xRALBJQQk%2FKyC%2Bx6Z8J5inmovtflrhOL0Cvm1mNRaQpVclNDAGYSzpOUamdK%2FlJE%2FQE%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260320T110721Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYTXKHTLAE%2F20260320%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=a4302a5660eef862d1845af92f4e5db2995cf0e523b59e15f99a3d72941f825a&hash=3fe0c26f74abb9f667c9d80ad4bf2664e1dd49869b1e0445ce629b1d60e03d53&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=0021999185901482&tid=spdf-8999359f-9965-4281-ad4e-09e6ad2782bf&sid=6631e4b18440084e6799914-00d95cf11814gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=02055c0a045455540c56&rr=9df440b17fc4631d&cc=gb

import numpy as np
from matplotlib import pyplot
from solver import make_solver

# Build 65 x 65 solver
run_65 = make_solver(nx=65, ny=65, ab2=True, implicit_diffusion=True)

# Build 97 x 97 solver for the higher Re cases
run_97 = make_solver(nx=97, ny=97, ab2=True, implicit_diffusion=True)

# Grid sizes from the paper:
# 65x65 for Re = 1, 100, 400
# 97x97 for Re = 1000, 2000, 5000
cases = [
    (1, run_65, '(a) Re = 1'),
    (100, run_65, '(b) Re = 100'),
    (400, run_65, '(c) Re = 400'),
    (1000, run_97, '(d) Re = 1000'),
    (2000, run_97, '(e) Re = 2000'),
    (5000, run_97, '(f) Re = 5000'),
]

fig, axes = pyplot.subplots(3, 2, figsize=(15, 10))

for ax, (re_val, run, title) in zip(axes.flat, cases):

    # 'run' is a function
    x, y, U_data, V_data, _ = run(re_val)

    U = U_data.T
    V = V_data.T

    # compute vector magnitude at each point
    # arrows point in direction of velocity vector (U,V). e.g if (U,V)=(-1,-1) then the arrow points down and left.
    mag = np.sqrt(U**2 + V**2)
    # where vel is zero, set magnitude to 1 to avoid division by zero when normalising arrows
    mag[mag == 0] = 1.0

    n = U.shape[1] # scale arrow size by grid size
    # normalise arrows, show direction only 
    ax.quiver(x, y, U/mag, V/mag, scale=n, scale_units='width', headwidth=2, headlength=2, headaxislength=2,
              width=0.002, color='k')
    
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

pyplot.tight_layout()
pyplot.savefig('figure5.png', dpi=150, bbox_inches='tight')
pyplot.show()
