// Separate translation unit: an out-of-line "library node" body. Without LTO the optimizer in
// the main TU cannot see through this, so it is a hard optimization barrier at each call site.
double fma_op(double x, double a, double b) { return x * a + b; }
