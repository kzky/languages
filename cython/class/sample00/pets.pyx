cdef class Parrot:

    cpdef describe(self):
        print("This parrot is resting.")

cdef class Norwegian(Parrot):

    cpdef describe(self):
        Parrot.describe(self)
        print("Lovely plumage!")

