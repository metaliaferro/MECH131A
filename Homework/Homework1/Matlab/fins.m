close all;
clear;

global AMIN L_OFFSET ODEOPT MINOPT;

AMIN = 1e-10;
RTOL = 1e-10;
ATOL = 1e-10;
TOL = 1e-10;
L_OFFSET = 1e-2;

ODEOPT = odeset('RelTol', RTOL, 'AbsTol', ATOL, 'NormControl', 'on');
MINOPT = optimset('Display', 'iter', 'MaxIter', 1000, 'MaxFunEvals', 1000, 'TolX', TOL, 'TolFun', TOL);

% Define constants
h = 10.0;  % W/m^2-K
k = 10.0;  % W/m-K
t = 0.5;   % Thickness

% Loop over different lengths and fin types
for L = [0.1, 0.18, 0.3, 0.56, 1.0, 1.8, 3.0]
    figure;
    for fin_type = {'rectangular', 'triangular', 'parabolic'}
        fin_type = fin_type{1};  % Extract string from cell

        % Get geometry and calculate m
        [Ac, dAcdx, dAsdx, Ap, As] = fin_geometry(0.0, t, L, fin_type);
        m = sqrt((h * L) / (k * Ap));

        % Get temperature profile along the fin
        [x, T, dTdx, error_num] = fin_profile_numerical(m, L, t, fin_type);

        % Compute convective losses
        q_c = convective_losses(x, t, L, h, T, fin_type);
        error_energy = q_c + k * dTdx(1);

        % Display results and plot temperature profiles
        fprintf('%s fin: x(end) = %.2f, Error = %.6f, Total Energy = %.6f\n', ...
                fin_type, x(end), error_num, q_c + k * t * dTdx(1));
        plot(x, T, 'DisplayName', fin_type, 'LineWidth', 1.5); hold on;
    end
    title(sprintf('mL = %.2f', m * L));
    ylim([-0.1, 1.1]);
    xlim([0, L])
    legend('show');
    hold off;
end

%%

function [Ac, dAcdx, dAsdx, Ap, As] = geometry_rectangular(x, t, L)
    global AMIN;
    % Ac, dAsdx, and As are given per unit width
    Ac = t;
    dAcdx = 0.0;
    dAsdx = 2.0;
    Ap = t * L;
    As = 2.0 * L;
    
    Ac = max(Ac, AMIN);
end

function [Ac, dAcdx, dAsdx, Ap, As] = geometry_triangular(x, t, L)
    global AMIN;
    % Ac, dAsdx, and As are given per unit width
    Ac = t * (1.0 - x / L);
    dAcdx = -t / L;
    dAsdx = 2.0 * sqrt((t / (2.0 * L))^2 + 1.0);
    Ap = t * L / 2.0;
    As = 2.0 * sqrt(L^2 + (t / 2.0)^2);
    
    Ac = max(Ac, AMIN);
end

function [Ac, dAcdx, dAsdx, Ap, As] = geometry_parabolic(x, t, L)
    global AMIN;
    % Ac, dAsdx, and As are given per unit width
    Ac = t * (1.0 - x / L).^2;
    dAcdx = -(2.0 * t * (L - x)) / L.^2;
    dAsdx = 2.0 * sqrt(1.0 + (t * (x - L) / L.^2).^2);
    Ap = L * t / 3.0;
    C1 = sqrt(1.0 + (t / L).^2);
    As = C1 * L + L.^2 * log(C1 + t / L) / t;
    
    Ac = max(Ac, AMIN);
end

function [Ac, dAcdx, dAsdx, Ap, As] = fin_geometry(x, t, L, fin_type)
    % Return geometry information based on fin type
    switch fin_type
        case 'rectangular'
            [Ac, dAcdx, dAsdx, Ap, As] = geometry_rectangular(x, t, L);
        case 'parabolic'
            [Ac, dAcdx, dAsdx, Ap, As] = geometry_parabolic(x, t, L);
        case 'triangular'
            [Ac, dAcdx, dAsdx, Ap, As] = geometry_triangular(x, t, L);
    end
end

function dydx = fin_equations(x, y, t, L, m, fin_type)
    % Solve the fin equation for the specified type of fin
    % y(1) = T, y(2) = dT/dx
    
    % Get geometry information and h/k
    [Ac, dAcdx, dAsdx, Ap, ~] = fin_geometry(x, t, L, fin_type);
    h_k = m^2 * Ap / L;

    dydx = zeros(2,1);
    % let dydx(1) = dT/dx = y(2)
    % let dydx(2) = d(dT/dx)/dx
    dydx(1) = y(2);
    dydx(2) = - (1.0 / Ac) * dAcdx * y(2) + (1.0 / Ac) * h_k * dAsdx * y(1);
end

function err = tip_error(dTdx_base, m, L, t, fin_type)
    global L_OFFSET ODEOPT;
    % Solve IVP using ode45 or similar solvers
    [~, res] = ode45(@(x, y) fin_equations(x, y, t, L, m, fin_type), ...
                     [0, (1.0 - L_OFFSET) * L], [1.0, dTdx_base], ODEOPT);
    
    T_end = res(end, 1);
    dTdx_end = res(end, 2);
    [~, ~, ~, Ap, ~] = fin_geometry((1.0 - L_OFFSET) * L, t, L, fin_type);
    
    % Compute error for optimization
    k_h = L / (m^2 * Ap);
    err = (T_end + k_h * dTdx_end)^2;
end

function [x_vals, T_vals, dTdx_vals, error] = fin_profile_numerical(m, L, t, fin_type)
    global L_OFFSET ODEOPT MINOPT;
    
    minres = fminsearch(@(dTdx_base) tip_error(dTdx_base, m, L, t, fin_type), -m, MINOPT);
    % minres = fsolve(@(dTdx_base) tip_error(dTdx_base, m, L, t, fin_type), -m, MINOPT);

    % Solve the IVP with the optimized boundary condition
    [x_vals, res] = ode45(@(x, y) fin_equations(x, y, t, L, m, fin_type), ...
                          [0, (1.0 - L_OFFSET) * L], [1.0, minres], ODEOPT);

    % Extract solution and compute error
    T_vals = res(:, 1);
    dTdx_vals = res(:, 2);
    error = T_vals(end) + dTdx_vals(end);   % This works as h = k
    
    % Correct negative values in temperature profile
    T_vals(T_vals < 0) = 0;
end

function Q_total = convective_losses(x, t, L, h, T, fin_type)
    % Numerically integrates the heat transfer from the surface of the fin

    % Get the surface area gradient dAs/dx at each x-location
    dAsdx = zeros(size(x));
    for i = 1:length(x)
        [~, ~, dAsdx(i), ~, ~] = fin_geometry(x(i), t, L, fin_type);
    end

    % Perform cumulative trapezoidal integration to get the total surface area
    As = zeros(size(x));
    As(1:end) = cumtrapz(x, dAsdx);
    [~, ~, ~, ~, As_check] = fin_geometry(L, t, L, fin_type);

    % Extract the surface area associated with each x-location
    dAs = diff(As);
    As_element = zeros(size(As));
    As_element(1) = 0.5 * dAs(1);
    As_element(end) = 0.5 * dAs(end);
    As_element(2:end-1) = 0.5 * dAs(1:end-1) + 0.5 * dAs(2:end);

    % Calculate the tip heat transfer for rectangular fin type
    Q_tip = h * t * T(end) * strcmp(fin_type, 'rectangular');

    % Return the total integrated heat transfer
    Q_total = sum(h * As_element .* T) + Q_tip;
end

