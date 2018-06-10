#!/usr/bin/ruby -s
# -*- Ruby -*-

# 《程序员的数学3:线性代数》配套源代码
# (原著：平岡和幸, 堀玄, 欧姆社, 2004. ISBN 4-274-06578-2)
# http://ssl.ohmsha.co.jp/cgi-bin/menu.cgi?ISBN=4-274-06578-2

# $Id: mymatrix.rb,v 1.13 2004/10/06 09:18:00 hira Exp $

# Copyright (c) 2004, HIRAOKA Kazuyuki <hira@ics.saitama-u.ac.jp>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice,this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
#    * Neither the name of the HIRAOKA Kazuyuki nor the names of its
#      contributors may be used to endorse or promote products derived
#      from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#########################################################

# 以下是矩阵的“加减乘运算”以及“通过LU分解求逆矩阵、解线性方程组”的程序代码。
# 本程序代码的目的在于演示笔算法的计算过程。
# 在真正的计算中，推荐大家使用 matrix.rb (Ruby 的标准库)等。

# 致广大 Ruby 用户:
#
# 非常抱歉。为了让更多的读者能理解算法，我们这里尽量避免了 Ruby 的代码风格。
#

# 致广大非 Ruby 用户:
#
# 请把下面的代码当作“伪代码”来读。
# 只要有任何主流程序设计语言的使用经验，应该就可以无障碍理解下面代码。
# （请读者千万不要误解，真正的 Ruby 语言, 可不是只能做做这些简单的工作而已。）
# 正如大家现在看到的，从“#”号开始一直到行尾，都是注释。
#
# 标记有“▼”的部分为依靠语言自身特性的部分。读者可以无视其中的内容。
# 请读者通过下面给出的例子确认使用方法。

#########################################################
# ▼Usage

def matrix_usage()
    name = File::basename $0
    print <<EOU
    #{name}: 用于矩阵计算的源程序
    (各种操作)
    #{name} -t=make   → 生成
    #{name} -t=print  → 显示（输出）
    #{name} -t=arith  → 和、数量乘法、积
    #{name} -t=op     → + - *
    #{name} -t=lu    → LU 分解
    #{name} -t=det    → 行列式
    #{name} -t=sol    → 线性方程组
    #{name} -t=inv    → 逆矩阵
    #{name} -t=plu    → LU 分解 (带有选主元的)
    EOU
end

def matrix_test(section)
    standalone = (__FILE__ == $0)  # 是否直接运行(并非从其他地方 load 进来)
    matched = (section == $t)  # -t= 后面的参数是否与 section 一致
    return (standalone and matched)
end

if (matrix_test(nil)) # 直接运行并且没有 -t 参数的情况下…
    matrix_usage()
end

#########################################################
# ▼向量、矩阵的生成与引用

# 引数の範囲チェックはサボリ

### 向量

class MyVector
    def initialize(n)
        @a = Array::new(n)
        for i in 0...n
            @a[i] = nil
        end
    end
    def [](i)
        return @a[i-1]
    end
    def []=(i, x)
        @a[i-1] = x
    end
    def dim
        return @a.length
    end
end

def make_vector(dim)
    return MyVector::new(dim)
end
def vector(elements)
    dim = elements.length
    vec = make_vector(dim)
    for i in 1..dim
        vec[i] = elements[i-1]
    end
    return vec
end
def vector_size(vec)
    return vec.dim
end
def vector_copy(vec)
    dim = vector_size(vec)
    new_vec = make_vector(dim)
    for i in 1..dim
        new_vec[i] = vec[i]
    end
    return new_vec
end

### 行列

class MyMatrix
    def initialize(m, n)
        @a = Array::new(m)
        for i in 0...m
            @a[i] = Array::new(n)
            for j in 0...n
                @a[i][j] = nil
            end
        end
    end
    def [](i, j)
        return @a[i-1][j-1]
    end
    def []=(i, j, x)
        @a[i-1][j-1] = x
    end
    def dim
        return @a.length, @a[0].length
    end
end

def make_matrix(rows, cols)
    return MyMatrix::new(rows, cols)
end
def matrix(elements)
    rows = elements.length
    cols = elements[0].length
    mat = make_matrix(rows, cols)
    for i in 1..rows
        for j in 1..cols
            mat[i,j] = elements[i-1][j-1]
        end
    end
    return mat
end
def matrix_size(mat)
    return mat.dim
end
def matrix_copy(mat)
    rows, cols = matrix_size(mat)
    new_mat = make_matrix(rows, cols)
    for i in 1..rows
        for j in 1..cols
            new_mat[i,j] = mat[i,j]
        end
    end
    return new_mat
end

### 例

if (matrix_test('make'))
    puts('- vector -')  # → 显示“- vector -”并换行
    
    puts('Make vector v = [2,9,4]^T, show v[2] and size of v.')
    v = make_vector(3)  # 生成 3 维列向量
    v[1] = 2
    v[2] = 9
    v[3] = 4
    puts(v[2])  # → 显示 9 并换行
    puts(vector_size(v))  # → 3 (维数)
    
    puts('Make vector w = [2,9,4]^T and show w[2].')
    w = vector([2,9,4])  # 同一个向量的另一种生成方法
    puts(w[2])  # → 9
    
    puts('Copy v to x and show x[2].')
    x = vector_copy(v)  # 复制
    puts(x[2])  # → 9
    puts('Modify x[2] and show x[2] again.')
    x[2] = 0
    puts(x[2])  # → 0
    puts('Original v[2] is not modified.')
    puts(v[2])  # → 9
    
    puts('- matrix -')
    
    puts('Make matrix A = [[2 9 4] [7 5 3]] and show a[2,1].')
    a = make_matrix(2, 3)  # 生成 2×3 矩阵
    a[1,1] = 2
    a[1,2] = 9
    a[1,3] = 4
    a[2,1] = 7
    a[2,2] = 5
    a[2,3] = 3
    puts(a[2,1])  # → 7
    puts('Show size of A.')
    rows, cols = matrix_size(a)  # 得到 a 的规模（行数、列数）
    puts(rows)  # → 2
    puts(cols)  # → 3
    
    puts('Make matrix B = [[2 9 4] [7 5 3]] and show b[2,1].')
    b = matrix([[2,9,4], [7,5,3]])  # 同一个矩阵的另一种生成方法
    puts(b[2,1])  # → 7
    
    puts('Copy A to C and show c[2,1].')
    c = matrix_copy(a)  # 复制
    puts(c[2,1])  # → 7
    puts('Modify c[2,1] and show c[2,1] again.')
    c[2,1] = 0
    puts(c[2,1])  # → 0
    puts('Original a[2,1] is not modified.')
    puts(a[2,1])  # → 7
end

#########################################################
# 显示向量和矩阵

# 定义显示向量的函数 vector_print。使用方法请参考下例。
def vector_print(vec)
    dim = vector_size(vec)
    for i in 1..dim  # 关于 i = 1, 2, ..., dim 的循环 (到 end 为止)
        printf('%5.4g ', vec[i])  # 要求显示位数至少为5位数，且只保留精度到小数点后第4位
        puts('')  # 换行
    end
    puts('')
end

def matrix_print(mat)
    rows, cols = matrix_size(mat)
    for i in 1..rows
        for j in 1..cols
            printf('%5.4g ', mat[i,j])
        end
        puts('')
    end
    puts('')
end

# 因为每次都写“vector_print”、“matrix_print”太长了，所以…
def vp(mat)
    vector_print(mat)
end
def mp(mat)
    matrix_print(mat)
end

### 例

if (matrix_test('print'))
    puts('Print vector [3,1,4]^T twice.')
    v = vector([3,1,4])
    vector_print(v)
    vp(v)
    puts('Print matrix [[2 9 4] [7 5 3]] twice.')
    a = matrix([[2,9,4], [7,5,3]])
    matrix_print(a)
    mp(a)
end

#########################################################
# 向量与矩阵的加法、数量乘法、积

### 向量

# 加法 (在向量 a 上加向量 b : a ← a+b) --- “#”以后为注释
def vector_add(a, b)       # 定义函数 (到 end 为止)
    a_dim = vector_size(a)   # 得到各个向量的维数
    b_dim = vector_size(b)
    if (a_dim != b_dim)      # 如果维数不相等的话… (到 end 为止)
        raise 'Size mismatch.' # 返回错误
    end
    # 从这里开始进入正题
    for i in 1..a_dim        # 循环 (到 end 为止): i = 1, 2, ..., a_dim
        a[i] = a[i] + b[i]     # 按分量相加
    end
end

# 数量乘法 (向量 vec 变成原来的 num 倍)
def vector_times(vec, num)
    dim = vector_size(vec)
    for i in 1..dim
        vec[i] = num * vec[i]
    end
end

### 矩阵

# 加法 (在矩阵 a 上加上矩阵 b : a ← a+b)
def matrix_add(a, b)
    a_rows, a_cols = matrix_size(a)
    b_rows, b_cols = matrix_size(b)
    if (a_rows != b_rows)
        raise 'Size mismatch (rows).'
    end
    if (a_cols != b_cols)
        raise 'Size mismatch (cols).'
    end
    for i in 1..a_rows
        for j in 1..a_cols
            a[i,j] = a[i,j] + b[i,j]
        end
    end
end

# 数量乘法 (矩阵 mat 变成原来的 num 倍)
def matrix_times(mat, num)
    rows, cols = matrix_size(mat)
    for i in 1..rows
        for j in 1..cols
            mat[i,j] = num * mat[i,j]
        end
    end
end

# 把矩阵 a 和向量 v 的乘积存入向量 r
def matrix_vector_prod(a, v, r)
    # 得到矩阵的规模
    a_rows, a_cols = matrix_size(a)
    v_dim = vector_size(v)
    r_dim = vector_size(r)
    # 确认是否满足矩阵乘积的定义
    if (a_cols != v_dim or a_rows != r_dim)
        raise 'Size mismatch.'
    end
    # 接下来进入正题。对于 a 的各行…
    for i in 1..a_rows
        # 将a与v的对应的元素（分量）相乘后累加
        s = 0
        for k in 1..a_cols
            s = s + a[i,k] * v[k]
        end
        # 计算结果存入 r
        r[i] = s
    end
end

# 矩阵 a 和矩阵 b 的乘积存入矩阵 r
def matrix_prod(a, b, r)
    # 得到矩阵的规模, 并确认满足乘积的定义
    a_rows, a_cols = matrix_size(a)
    b_rows, b_cols = matrix_size(b)
    r_rows, r_cols = matrix_size(r)
    if (a_cols != b_rows or a_rows != r_rows or b_cols != r_cols)
        raise 'Size mismatch.'
    end
    # 从这里开始进入正题。关于 a 的各行、b 的各列…
    for i in 1..a_rows
        for j in 1..b_cols
            # 将 a 和 b 对应的元素相乘后累加
            s = 0
            for k in 1..a_cols
                s = s + a[i,k] * b[k,j]
            end
            # 将结果存入 r
            r[i,j] = s
        end
    end
end

### 例

if (matrix_test('arith'))
    puts('- vector -')
    
    v = vector([1,2])
    w = vector([3,4])
    
    c = vector_copy(v)
    vector_add(c,w)
    puts('v, w, v+w, and 10 v')
    vp(v)
    vp(w)
    vp(c)
    
    c = vector_copy(v)
    vector_times(c,10)
    vp(c)
    
    puts('- matrix -')
    
    a = matrix([[3,1], [4,1]])
    b = matrix([[10,20], [30,40]])
    
    c = matrix_copy(a)
    matrix_add(c, b)
    puts('A, B, A+B, and 10 A')
    mp(a)
    mp(b)
    mp(c)
    
    c = matrix_copy(a)
    matrix_times(c, 10)
    mp(c)
    
    r = make_vector(2)
    matrix_vector_prod(a, v, r)
    puts('A, v, and A v')
    mp(a)
    vp(v)
    vp(r)
    
    r = make_matrix(2,2)
    matrix_prod(a, b, r)
    puts('A, B, and A B')
    mp(a)
    mp(b)
    mp(r)
end

#########################################################
# ▼写成形如 a + b 的形式

class MyVector
    def +(vec)
    c = vector_copy(self)
    vector_add(c, vec)
    return c
end
def -@()  # 一元运算“-”
c = vector_copy(self)
vector_times(c, -1)
return c
end
def -(vec)
return self + (- vec)
end
def *(x)
dims = vector_size(self)
if (dims == 1)
    return x * self[1]
    elsif x.is_a? Numeric
    c = vector_copy(self)
    vector_times(c, x)
    return c
    else
    raise 'Type mismatch.'
end
end
def coerce(other)
    if other.is_a? Numeric
        return vector([other]), self
        else
        raise 'Unsupported type.'
    end
end
end

class MyMatrix
    def +(mat)
    c = matrix_copy(self)
    matrix_add(c, mat)
    return c
end
def -@()  # 一元运算“-”
c = matrix_copy(self)
matrix_times(c, -1)
return c
end
def -(mat)
return self + (- mat)
end
def *(x)
rows, cols = matrix_size(self)
if (rows == 1 and cols == 1)
    return x * self[1,1]
    elsif x.is_a? Numeric
    c = matrix_copy(self)
    matrix_times(c, x)
    return c
    elsif x.is_a? MyVector
    r = make_vector(cols)
    matrix_vector_prod(self, x, r)
    return r
    elsif x.is_a? MyMatrix
    x_rows, x_cols = matrix_size(x)
    r = make_matrix(rows, x_cols)
    matrix_prod(self, x, r)
    return r
    else
    raise 'Type mismatch.'
end
end
def coerce(other)
    if other.is_a? Numeric
        return matrix([[other]]), self
        else
        raise 'Unsupported type.'
    end
end
end

### 例

if (matrix_test('op'))
    puts('- vector -')
    x = vector([1,2])
    y = vector([3,4])
    puts('x, y')
    vp(x)
    vp(y)
    puts('x+y, -x, y-x, x*10, 10*x')
    vp(x + y)
    vp(- x)
    vp(y - x)
    vp(x * 10)
    vp(10 * x)
    
    puts('- matrix -')
    a = matrix([[3,1], [4,1]])
    b = matrix([[10,20], [30,40]])
    puts('A, B')
    mp(a)
    mp(b)
    puts('A+B, -A, B-A, A*10, 10*A, A*B')
    mp(a + b)
    mp(- a)
    mp(b - a)
    mp(a * 10)
    mp(10 * a)
    mp(a * b)
    puts('A, x, and A*x')
    mp(a)
    vp(x)
    vp(a * x)
end

#########################################################
# LU 分解 (没有选主元)

# LU 分解 (没有选主元).
# 可能有些对于会出现“除0”错误。
# 分解结果用原变量 mat 存储(左下部分为 L, 右上部分为 U)。
def lu_decomp(mat)
    rows, cols = matrix_size(mat)
    # 令 s 为行数 (rows) 和列数 (cols) 中较小的一个
    if (rows < cols)
        s = rows
        else
        s = cols
    end
    # 从这里开始进入正题
    for k in 1..s
        # 本阶段的 mat 如下所示 (u、l 分别为 U、L 的完成部分，r 为剩余部分)
        #     u u u u u u
        #     l u u u u u
        #     l l r r r r  ←第 k 行
        #     l l r r r r
        #     l l r r r r
        # 【甲】 这时 U 的第 k 行还未完成，不需要对这部分做任何操作。
        # 【乙】 计算 L 的第 k 列
        # 为了减少除法运算次数，这里用到了一个小技巧。
        x = 1.0 / mat[k,k]  # (如果 mat[k,k] 为 0，这里就会遇到“除0”错误。)
        for i in (k+1)..rows
            mat[i,k] = mat[i,k] * x  # 本质上就是 mat[i,k] / mat[k,k]
        end
        # 【丙】 更新未完成部分
        for i in (k+1)..rows
            for j in (k+1)..cols
                mat[i,j] = mat[i,j] - mat[i,k] * mat[k,j]
            end
        end
    end
end

# 将 LU 分解的结果分别存入两个矩阵 L, U
def lu_split(lu)
    rows, cols = matrix_size(lu)
    # 令 r 为行数和列数中较小的一个
    if (rows < cols)
        r = rows
        else
        r = cols
    end
    # L 的大小为 rows×r, R 的大小为 r×cols
    lmat = make_matrix(rows, r)
    umat = make_matrix(r, cols)
    # 求出 L
    for i in 1..rows
        for j in 1..r
            if (i > j)
                x = lu[i,j]
                elsif (i == j)  # else if
                x = 1
                else
                x = 0
            end
            lmat[i,j] = x
        end
    end
    # 求出 R
    for i in 1..r
        for j in 1..cols
            if (i > j)
                x = 0
                else
                x = lu[i,j]
            end
            umat[i,j] = x
        end
    end
    return [lmat, umat]  # 返回 lmat 和 umat 作为一组返回值
end

### 例

if (matrix_test('lu'))
    a = matrix([[2,6,4], [5,7,9]])
    c = matrix_copy(a)
    lu_decomp(c)
    l, u = lu_split(c)
    puts('A, L, U, and L U')
    mp(a)
    mp(l)
    mp(u)
    mp(l * u)  # 等于 a
end

#########################################################
# 行列式

# 行列式（原来存储矩阵的变量会被破坏掉)
def det(mat)
    # 确认矩阵为方阵
    rows, cols = matrix_size(mat)
    if (rows != cols)
        raise 'Not square.'
    end
    # 从这里开始进入正题。进行 LU 分解…
    lu_decomp(mat)
    # 返回 U 的对角元素的乘积
    x = 1
    for i in 1..rows
        x = x * mat[i,i]
    end
    return x
end

### 例

if (matrix_test('det'))
    a = matrix([[2,1,3,2], [6,6,10,7], [2,7,6,6], [4,5,10,9]])
    puts('A and det A = -12')
    mp(a)
    puts det(a)  # → -12
end

#########################################################
# 线性方程组

# 解方程 A x = y (A: 方阵, y: 向量)
# 解的过程中 A 会被破坏，解直接保存在 y 中。
def sol(a, y)
    # 省略了矩阵规模的确认步骤
    # 首先进行 LU 分解
    lu_decomp(a)
    # 封装，交给后面了
    sol_lu(a, y)
end

# (接上) 解方程 L U x = y。解保存在 y 中。
# 其中 L, U 为方阵 A 的 LU 分解 (两个矩阵保存在同一个矩阵变量中)
def sol_lu(lu, y)
    # 得到 y 的维数
    n = vector_size(y)
    # 解 L z = y，解 z 保存在 y 中
    sol_l(lu, y, n)
    # 解 U x = y (其中存储的是 z 的值)，解 x 保存在 y 中
    sol_u(lu, y, n)
end

# (接上) 解 L z = y。解 z 保存在 y 中。n 为 y 的维数。
# 其中 L, U 为方阵 A 的 LU 分解 (两个矩阵保存在同一个矩阵变量中)
def sol_l(lu, y, n)
    for i in 1..n
        # 计算出 z[i] = y[i] - L[i,1] z[1] - ... - L[i,i-1] z[i-1]
        # 将已求出的解 z[1], ..., z[i-1] 全部保存入 y[1], ..., y[i-1]
        for j in 1..(i-1)
            y[i] = y[i] - lu[i,j] * y[j]  # 实际为 y[i] - L[i,j] * z[j]
        end
    end
end

# (接上) 解 U x = y。解 x 保存入 y。 n 为 y 的维数。
# 其中 L, U 为方阵 A 的 LU 分解 (两个矩阵保存在同一个矩阵变量中)
def sol_u(lu, y, n)
    # 按照 i = n, n-1, ..., 1 的逆序进行处理
    #   ※ 慎重起见，声明一则:
    #   请不要误认为Ruby语言本身就是这么笨拙的。
    #   为了让不熟悉Ruby的读者也可以顺利的理解算法，所以没有采用方便的功能和函数。
    for k in 0..(n-1)
        i = n - k
        # 计算出 x[i] = (y[i] - U[i,i+1] x[i+1] - ... - U[i,n] x[n]) / U[i,i]
        # 将已求出的解 x[i+1], ..., x[n] 全部保存入 y[i+1], ..., y[n]
        for j in (i+1)..n
            y[i] = y[i] - lu[i,j] * y[j]  # 实际为 y[i] - U[i,j] * x[j]
        end
        y[i] = y[i] / lu[i,i]
    end
end

### 例

if (matrix_test('sol'))
    a = matrix([[2,3,3], [3,4,2], [-2,-2,3]])
    c = matrix_copy(a)
    y = vector([9,9,2])
    puts('A, y, and solution x of A x = y.')
    mp(a)
    vp(y)
    sol(c, y)
    vp(y)
    puts('A x')
    vp(a*y)
end

#########################################################
# 逆矩阵

# 返回逆行列。原来的矩阵会被破坏掉。
def inv(mat)
    rows, cols = matrix_size(mat)
    # 确认矩阵为方阵
    rows, cols = matrix_size(mat)
    if (rows != cols)
        raise 'Not square.'
    end
    # 准备保存结果用的变量。初始化为单位矩阵。
    ans = make_matrix(rows, cols)
    for i in 1..rows
        for j in 1..cols
            if (i == j)
                ans[i,j] = 1
                else
                ans[i,j] = 0
            end
        end
    end
    # 从这里开始进入正题。进行 LU 分解…
    lu_decomp(mat)
    for j in 1..rows
        # 将 ans 的各列视为线性方程组的 A x = y 的右边，进行求解。
        #   ※ 实际上恰当的做法应该是把 ans 的各列直接剥离出来直接传入 sol_lu 进行计算，
        #   不过这种方法是与语言本身特性有关的。没办法只能专门写成下面这样：
        #   (1)复制, (2)计算, (3)将结果写回去。
        v = make_vector(cols)
        for i in 1..cols
            v[i] = ans[i,j]
        end
        sol_lu(mat, v)
        for i in 1..cols
            ans[i,j] = v[i]
        end
    end
    return(ans)
end

if (matrix_test('inv'))
    a = matrix([[2,3,3], [3,4,2], [-2,-2,3]])
    c = matrix_copy(a)
    b = inv(c)
    puts('A and B = inverse of A.')
    mp(a)
    mp(b)
    puts('A B and B A')
    mp(a*b)
    mp(b*a)
end

#########################################################
# LU 分解 (有选主元)
# 将结果保存在原变量 mat 中, 返回值为选主元表 (向量 p)
#
# 结果为，形如
# A' = L U (A' 为 A 进行行交换之后的矩阵, L 为上三角阵, U 为下三角阵) 的分解。
# A' 的第 i 行是原矩阵 A 的第 p[i] 行。
# 可以通过 p_ref(mat, i, j, p) 得到 L (i>j) 或 U (i<=j)的 i,j 元素。
def plu_decomp(mat)
    rows, cols = matrix_size(mat)
    # 准备好选主元表，用于记录选主元之后的矩阵的各行和原矩阵的各行之间的对应关系。
    # 避免直接对 mat[i,j] 进行直接操作，而必须通过函数 p_ref(值引用), p_set(值変更)来访问选主元之后的矩阵。
    # 这样一来，前面的 lu_decomp 代码就可以复用了。
    p = make_vector(rows)
    for i in 1..rows
        p[i] = i  # 选主元表 的初始值为“第 i 行换到第 i 行”。
    end
    # 令 s 为行数 (rows) 和列数 (cols) 中较小的一个
    if (rows < cols)
        s = rows
        else
        s = cols
    end
    # 这里开始进入正题
    for k in 1..s
        # 首先进行选主元操作
        p_update(mat, k, rows, p)
        # 从这里开始，仅仅就是在重复 lu_decomp 的过程
        #   mat[i,j] → p_ref(mat, i, j, p)
        #   mat[i,j] = y → p_set(mat, i, j, p, y)
        # 【甲】 这时 U 的第 k 行还未完成，不需要对这部分做任何操作。
        # 【乙】 计算 L 的第 k 列
        x = 1.0 / p_ref(mat, k, k, p)
        for i in (k+1)..rows
            y = p_ref(mat, i, k, p) * x
            p_set(mat, i, k, p, y)
        end
        # 【丙】 更新未完成部分
        for i in (k+1)..rows
            x = p_ref(mat, i, k, p)
            for j in (k+1)..cols
                y = p_ref(mat, i, j, p) - x * p_ref(mat, k, j, p)
                p_set(mat, i, j, p, y)
            end
        end
    end
    # 返回选主元表。
    return(p)
end

# 进行选主元
# 具体来说就是将第 k 列未处理部分中绝对值最大的分量换到第 k 行上。
def p_update(mat, k, rows, p)
    # 在所有候选(第 k 列的未处理部分)分量中，找到“冠军元”（绝对值最大的分量)
    max_val = -777  # 最弱的第一代冠军，谁也赢不了。
    max_index = 0
    for i in k..rows
        x = abs(p_ref(mat, i, k, p))
        if (x > max_val)  # 如果冠军败了话
            max_val = x
            max_index = i
        end
    end
    # 将当前行(k)和冠军行(max_index)进行互换
    pk = p[k]
    p[k] = p[max_index]
    p[max_index] = pk
end

# 返回选主元后的矩阵的 (i,j) 元素的值
def p_ref(mat, i, j, p)
    return(mat[p[i], j])
end

# 将选主元后的矩阵的 (i,j) 元素的値变更为 val
def p_set(mat, i, j, p, val)
    mat[p[i], j] = val
end

# ▼绝对值(写成函数的形式)
def abs(x)
    return(x.abs)
end

# 将 LU 分解的结果分成两个矩阵 L, U 来存储
def plu_split(lu, p)
    rows, cols = matrix_size(lu)
    # 令 r 为行数和列数中较小的一个
    if (rows < cols)
        r = rows
        else
        r = cols
    end
    # L 的大小为 rows×r, R 的大小为 r×cols
    lmat = make_matrix(rows, r)
    umat = make_matrix(r, cols)
    # 求出 L
    for i in 1..rows
        for j in 1..r
            if (i > j)
                x = p_ref(lu, i, j, p)
                elsif (i == j)  # else if
                x = 1
                else
                x = 0
            end
            lmat[i,j] = x
        end
    end
    # 求出 R
    for i in 1..r
        for j in 1..cols
            if (i > j)
                x = 0
                else
                x = p_ref(lu, i, j, p)
            end
            umat[i,j] = x
        end
    end
    return [lmat, umat]  # 返回 lmat 和 umat 作为一组返回值
end

### 例

if (matrix_test('plu'))
    a = matrix([[2,6,4], [5,7,9]])
    c = matrix_copy(a)
    p = plu_decomp(c)
    l, u = plu_split(c, p)
    puts('A, L, U, and pivot table')
    mp(a)
    mp(l)
    mp(u)
    vp(p)
    puts('L U')
    mp(l * u)
end

#########################################################
# ▼最后的操作检查
# 与 matrix.rb 得到的结果进行对照比较。
# 后面不需要读者费心了。（解开封印）

if ($c)
    require 'matrix'
    $eps = 1e-10
    class MyVector
    def to_a
    @a
end
end
class MyMatrix
    def to_a
        @a
    end
end
def rmat(a)
    Matrix.rows a
end
def to_array_or_number(x)
    [Array, Matrix, Vector, MyVector, MyMatrix].find{|c| x.is_a? c} ? x.to_a : x
end
def aeq?(x, y)
    x = to_array_or_number x
    y = to_array_or_number y
    if x.is_a? Numeric
        y.is_a? Numeric and (x - y).abs < $eps
        elsif x.is_a? Array
        y.is_a? Array and
        x.size == y.size and
        not (0 ... x.size).map{|i| aeq? x[i], y[i]}.member? false
        else
        raise 'Bad type.'
    end
end
def rand_ary1(n)
    (1..n).map{|i| rand - 0.5}
end
def rand_ary2(m,n)
    (1..m).map{|i| rand_ary1 n}
end
def check_matmul(l,m,n)
    a = rand_ary2 l, m
    b = rand_ary2 m, n
    aeq? rmat(a) * rmat(b), matrix(a) * matrix(b)
end
def check_det(n)
    a = rand_ary2 n, n
    aeq? rmat(a).det, det(matrix(a))
end
def check_inv(n)
    a = rand_ary2 n, n
    aeq? rmat(a).inv, inv(matrix(a))
end
def check(label, repeat, proc)
    (1..repeat).each{|t| raise "#{label}" if !proc.call}
    puts "#{label}: ok"
end
[
['matmul', 100, lambda{check_matmul 6,5,4}],
['det', 100, lambda{check_det 7}],
['inv', 100, lambda{check_det 7}],
['aeq?', 1,
lambda{
    ![
    # all pairs must be !aeq?
    [3, 3.14],
    [Vector[3], 3],
    [3, vector([3])],
    [Vector[1,2,3], vector([1,2,3,4])],
    [Vector[1,2,3,4], vector([1,2,3])],
    [Vector[1.1,2.2,3.3], vector([1.1,2.2000001,3.3])],
    [rmat([[1,2,3], [4,5,6]]), matrix([[1,2,3], [4.0000001,5,6]])],
    ].map{|a| !aeq?(*a)}.member? false}],
['----------- All Tests -----------', 1, lambda{true}],
['This must fail. OK if you see an error.', 1, lambda{aeq? 7.77, 7.76}],
].each{|a| check *a}
end
