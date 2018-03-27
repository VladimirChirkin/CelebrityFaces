# distutils: language=c++
# distutils: sources=annoy.cpp
# distutils: extra_compile_args= -std=c++11 -fopenmp
# distutils: extra_link_args= -fopenmp

from cython.operator cimport dereference as deref
from cpython.array cimport array
cimport numpy as cnp
import numpy as np

cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        vector() except +
        void push_back(T&)
        T operator[](int i)
        int size()

cdef extern from "annoy.h":
    cdef cppclass FeatureVector:
        FeatureVector()
        void PushBack(double feature)

    cdef cppclass AnnoyForest:
        AnnoyForest(int node_size, int n_trees)
        void Fit(vector[FeatureVector]& embeddings)
        vector[int] Find(FeatureVector& emb, int n_search)

cdef class Annoy:
    cdef AnnoyForest* _thisptr

    def __cinit__(self, node_size, n_trees):
        self._thisptr = new AnnoyForest(node_size, n_trees)

    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr

    cpdef void fit(self, double[:, :] mv):
        cdef vector[FeatureVector] inp
        cdef int i, j, cur = 0
        for i in range(len(mv)):
            inp.push_back(FeatureVector())
            for j in range(len(mv[0])):
                inp[cur].PushBack(mv[i, j])
            cur += 1
        self._thisptr.Fit(inp)

    cdef vector[int] _find(self, double[:] mv, int n_search):
        cdef FeatureVector inp
        cdef double elem
        for elem in mv:
            inp.PushBack(elem)
        return self._thisptr.Find(inp, n_search)

    def find(self, emb, n_search):
        cdef vector[int] answer = self._find(emb, n_search)
        arranswer = np.zeros((answer.size(),), int)
        for i in range(len(arranswer)):
            arranswer[i] = answer[i]
  
        return arranswer

