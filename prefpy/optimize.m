% Filename: optimize.m
% Original Author: Andrew Ning (https://bitbucket.org/andrewning/)
% Original Version: https://bitbucket.org/mdolab/pyoptsparse/pull-requests/13/an-example-bridge-to-fmincon/diff
% Modified By: Peter Piech
% Date: 1/19/2016

function [xopt, fopt, exitflag] = optimize(funcname, args, x0, ...
    A, b, Aeq, beq, lb, ub, options)

    % set options
    opts = optimoptions('fmincon');
    names = fieldnames(options);
    for i = 1:length(names)
        opts = optimoptions(opts, names{i}, options.(names{i}));
    end

    % run fmincon
    [xopt, fopt, exitflag] = fmincon(@(x) obj(x, args), x0, A, b, Aeq, beq, lb, ub, [], opts);

    function J = obj(x, args)
        J = py.(funcname)(x, args);
    end
end
