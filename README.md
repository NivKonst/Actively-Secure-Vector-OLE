# Actively-Secure-Vector-OLE

Vector Oblivious Linear Evaluation (VOLE) over a finite field F is a two-party functionality that takes from a sender a pair of vectors (a,b) of length w each, and delivers to a receiver who holds a scalar x, the value xa+b. VOLE is a useful vectorized extension of Oblivious Linear Evaluation (where a and b are scalars) and it can be viewed as the arithmetic analog of String Oblivious Transfer. VOLE has several important applications in secure multiparty computation and zero-knowledge interactive proofs. 

In the passive setting, Applebaum et al. (Crypto 2017) constructed a protocol that only makes a constant (amortized) number of field operations per VOLE entry. This protocol uses the underlying field F as a black box and makes black-box use of (standard) oblivious transfer. In our work we present an actively-secure variant of this protocol that achieves, for the first time, all the above features and adds only a minor overhead in computation and communication.

We present a Python implementation and suggest several practical and theoretical optimizations. Our most efficient variant can achieve an asymptotic rate of 1/4 (i.e., for vectors of length w we send roughly 4w elements of F), which is only slightly worse than the passively-secure variant of the protocol whose rate is 1/3. The protocol seems to be practically competitive over fast networks, even for relatively small fields F and relatively short vectors. Specifically, our VOLE protocol has 3 rounds, and even for 10K-long vectors, it has an amortized cost per entry of less than 4 OT's and less than 300 arithmetic operations. Most of these operations (about 200) can be pre-processed locally in an offline non-interactive phase. (Better constants can be obtained for longer vectors.) 

More details about our work can be found at:

https://eprint.iacr.org/2023/270
https://www.iacr.org/cryptodb/data/paper.php?pubkey=33002
https://link.springer.com/chapter/10.1007/978-3-031-30617-4_7

A presentation about of our work from Eurocrypt 2023 can be found at: https://youtu.be/Aq_btPAt1T4?si=_OXNyWpcT83SZ1RB&t=1426


## Installation

In order to run this project, use python 3.7 and to make sure that you have all the needed packages. Use the pip module and install the packages from the requirements.txt file:

`python -m pip install -r requirements.txt`
