import numpy as np
cimport numpy as np

cpdef rainflow(np.ndarray extrema):
    cdef list residuen_vector = []

    cdef int i
    cdef double a
    cdef double b
    cdef double c

    points = extrema.size / 2

    # table_obj = hdf5_file.table_cache[table]
    cdef np.ndarray array = np.zeros((int(points),2), dtype=np.float64)

    cdef int len_residuen_vector = 0
    cdef int array_index = 0
    for x in extrema:
        residuen_vector.append(x)
        len_residuen_vector += 1

        if len_residuen_vector < 3:
            continue

        i = 0
        while i + 2 < len_residuen_vector:
            a = residuen_vector[i]
            b = residuen_vector[i + 1]
            c = residuen_vector[i + 2]

            if (a > b and c >= a) or (a < b and c <= a):
                value = (a, b)
                # table_obj.append(value)
                array[array_index] = value
                array_index += 1
                residuen_vector.pop(i + 1)
                residuen_vector.pop(i)
                i = 0
                len_residuen_vector -= 2
            else:
                i += 1

    return array[0:array_index], residuen_vector