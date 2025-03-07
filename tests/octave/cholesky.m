% Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
M = [2 -1 0; -1 2 -1 ; 0 -1 2]
n = length( M );
L = zeros( n, n );
for i=1:n
   L(i, i) = sqrt(M(i, i) - L(i, :)*L(i, :)');
   for j=(i + 1):n
      L(j, i) = (M(j, i) - L(i,:)*L(j ,:)')/L(i, i);
   end
end
