function [val, guess, varsOpt] = compare(problem)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    [guess, r] = GuessSolution(problem);
    [rhoOpts, varsOpt, Xopts] = MaxRho(problem);
    syms X;
    dif = simplify(guess - varsOpt);
    val = all(dif == 0);
end

