#
#     This file is part of CasADi.
# 
#     CasADi -- A symbolic framework for dynamic optimization.
#     Copyright (C) 2010 by Joel Andersson, Moritz Diehl, K.U.Leuven. All rights reserved.
# 
#     CasADi is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
# 
#     CasADi is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
# 
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
# 
# 
#! jacSparsity
#!======================
from casadi import *
from numpy import *
import casadi as c
from pylab import spy, show

#! We construct a simple SX expression
x = ssym("x",40)
y = x[:-2]-2*x[1:-1]+x[2:]

#! Let's see what the first 5 entries of y look like
print y[:5]

#! Next, we construct a function
f = SXFunction([x],[y])
f.init()

#! And we visualize the sparsity of the jacobian
spy(f.jacSparsity())

show()

