#! /usr/bin/env python

# Tema 1 Retele Neuronale
# Bejan Irina [B3]

import copy
import sys
import re
import numpy as np


_COEF = r"[+-]?\d*"
REGEX = r"({coef}x|)({coef}y|)({coef}z|)=({coef})".format(coef=_COEF)


def compute_minors(mat):
    minors = []
    for i in range(3):
        mrow = []
        for j in range(3):
            det = []
            for im in range(3):
                if im == i:
                    continue
                det_row = []
                for jm in range(3):
                    if jm == j:
                        continue
                    det_row.append(mat[im][jm])
                det.append(det_row)
            det = det[0][0] * det[1][1] - det[0][1] * det[1][0]
            mrow.append(det)
        minors.append(mrow)
    
    return minors


def apply_cofactors(minors):
    for i in range(3):
        for j in range(3):
            minors[i][j] = minors[i][j] * pow(-1, i + j)
    return minors


def transpose(mat):
    transposed = copy.deepcopy(mat)
    for i in range(3):
        for j in range(3):
            transposed[j][i] = mat[i][j]

    return transposed


def compute_det(mat, minors):
    det = 0
    for i in range(3):
        det += mat[0][i] * minors[i][0]
    return det

def compute_values(inverse, vec):
    values = []
    for i in range(3):
        t = sum([a * b for a, b in zip(inverse[i], vec)])
        values.append(t)

    return values

def solve_basic(mat, vec):
    minors = compute_minors(mat)
    minors = apply_cofactors(minors)
    A = transpose(minors)
    det = compute_det(mat, A)
    if not det:
        print("Determinant is null!")
        return None
    
    for i in range(3):
        for j in range(3):
            A[i][j] *= (1.0/det)

    return compute_values(A, vec)


def solve_numpy(mat, vec):
    mat = np.array(mat)
    vec = np.array(vec)
    try:
        return list(np.linalg.inv(mat).dot(vec))
    except Exception:
        return None


def parse(fname):
    """Parses the input file"""
    with open(fname, "rb") as stream:
        data = stream.read()
    mat = []
    vec = []
    for line in data.strip().splitlines():
        line = line.strip().replace(" ", "")
        match = re.match(REGEX, line)
        if not match:
            return None
        x, y, z, r = match.groups()
        cfs = [x, y, z]
        for idx, cf in enumerate(cfs):
            if not cf:
                cfs[idx] = cf
                continue
            cf = cf.rstrip("xyz")
            if not cf or not cf[-1].isdigit():
                cf += "1"
            cfs[idx] = cf
        mat.append(list(map(lambda arg: float(arg) if arg else 0.0, cfs)))
        vec.append(float(r) or 0.0)
    return mat, vec


def main(argv):
    if len(argv) != 2:
        print("[i] Usage: {} {}".format(argv[0], "FILE"))
        return 1
    
    ret = parse(argv[1])
    if not ret:
        print("Invalid expression!")
        return 2

    for solve in [solve_basic, solve_numpy]:
        res = solve(*ret)
        if not res:
            print("Couldn't solve the equation!")
            return 3

        print("x={0}, y={1}, z={2}".format(*res))
    return 0


if __name__ == "__main__":
    main(sys.argv)
