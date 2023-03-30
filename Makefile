# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

CXX ?= g++
OUT ?= build/libpoptorch_experimental_addons.so
OBJDIR ?= $(dir $(OUT))obj

CXXFLAGS = -Wall -Wextra -Werror -std=c++17 -O2 -g -fPIC -DONNX_NAMESPACE=onnx
LIBS = -lpoplar -lpopart -lpopops -lpopsparse -lpoputil -lgcl

OBJECTS = $(OBJDIR)/static_spmm.o $(OBJDIR)/autograd_proxy.o $(OBJDIR)/replicatedallreducetp.o $(OBJDIR)/distance_matrix.o $(OBJDIR)/replicatedalltoall.o $(OBJDIR)/replicatedalltoallx.o

# Rules

.DEFAULT_GOAL := $(OUT)

$(OBJECTS): $(OBJDIR)/%.o: poptorch_experimental_addons/cpp/%.cpp
	@mkdir -p $(@D)
	$(CXX) -c $(CXXFLAGS) $< -o $@

$(OUT): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -shared $^ -o $@ -Wl,--no-undefined $(LIBS)
