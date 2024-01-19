% k-Wave simulation of a convex, circular array transducer
% Author: Mikael Haji

clearvars;

% =========================================================================
%   PARAMETERS - define the params and init everything
% =========================================================================

% Transducer array parameters
radii = [10e-3, 20e-3, 30e-3];  % radii of the concentric circles [10mm, 20mm, 30mm]
elements_per_circle = [6, 12, 18];  % number of elements per circle
total_elements = sum(elements_per_circle);
element_width = 1e-3;  % element width [m] (how big is it width wise)
element_length = 10e-3;  % element length [m] '' length

% Source parameters
source_f0 = 1e6;  % 1 mill Hz, source frequency [Hz], the frequency of the ultrasonic wave emitted by the actual transducer element
source_cycles = 5;  % number of cycles in the tone burst, 5 cycles in a particular burst of sound waves

% Medium properties
c0 = 1500;  % sound speed [m/s], speed of the sound in the medium (1500 usually for water or soft tissues)
rho0 = 1000;  % density [kg/m^3] (density of water)



% Time parameters
ppw = 3;  % points per wavelength (per a given wavelenth how many points fall under it)
t_end = 60e-6;  % total simulation time [s]


% Grid Point Spacing: spacing between the grid points in the simulation,
% dictates resolution
% Grid parameters (adjusted for 3D array)
dx = c0 / (ppw * source_f0);  % Adjust grid spacing
dy = dx;      % Grid point spacing in y direction [m]
dz = dx;      % Grid point spacing in z direction [m]

grid_size_x = max(70e-3, max(radii) * 2.5);  % Increased grid size for safety
grid_size_y = max(70e-3, max(radii) * 2.5);
grid_size_z = max(50e-3, max(radii) * 2.5);


% =========================================================================
% CREATE ARRAY
% =========================================================================

fprintf('\n\nCreating Array......\n')

% Create empty kWaveArray
karray = kWaveArray();

% Define the focal point at the geometric center
% focal_point = [0, 0, 0];

% Example focal point settings
focal_point_x = 0;  % X-coordinate of the focal point
focal_point_y = 0;  % Y-coordinate of the focal point
focal_point_z = 40e-3;  % Z-coordinate of the focal point, 20 mm upward

% Define the focal point
focal_point = [focal_point_x, focal_point_y, focal_point_z];


% Add elements to each circle with 3D positioning
for circle = 1:length(radii)
    radius = radii(circle);
    num_elements = elements_per_circle(circle);
    for ind = 1:num_elements
        angle = 2 * pi * (ind - 1) / num_elements;
        x_pos = radius * cos(angle);
        y_pos = radius * sin(angle);
        z_pos = radius * sin(angle / 2);
        
        % Calculate the direction vector from element to the focal point
        direction = focal_point - [x_pos, y_pos, z_pos];

        % Calculate pitch (rotation around x-axis) and yaw (rotation around y-axis)
        pitch = atan2(-direction(3), sqrt(direction(1)^2 + direction(2)^2)); % negative z to point upwards
        yaw = atan2(direction(2), direction(1));

        % Assuming roll (rotation around z-axis) is zero
        roll = 0;

        theta = [pitch, yaw, roll];
        karray.addRectElement([x_pos, y_pos, z_pos], element_width, element_length, theta);

    end
end


% After creating the array with karray.addRectElement
% Get positions of all elements
element_positions = karray.getElementPositions();

% Count the total number of elements
num_elements = size(element_positions, 2); % Second dimension gives the number of columns/elements

% Display the total number of elements
fprintf('Total number of elements in the array: %d\n', num_elements);

% Visualize element positions
figure;
scatter3(element_positions(1, :), element_positions(2, :), element_positions(3, :), 'filled');
xlabel('X Position (m)');
ylabel('Y Position (m)');
zlabel('Z Position (m)');
title('Visualization of Transducer Element Positions');
axis equal; % This ensures equal scaling for all axes
grid on;

% Visualize orientation vectors
figure;
hold on;

% Plotting the focal point
plot3(focal_point(1), focal_point(2), focal_point(3), 'r*', 'MarkerSize', 10, 'LineWidth', 2);
text(focal_point(1), focal_point(2), focal_point(3), ' Focal Point', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');

% Retrieve element positions
element_positions = karray.getElementPositions();

% Plot each element and its direction vector
for i = 1:size(element_positions, 2)
    % Element position
    x = element_positions(1, i);
    y = element_positions(2, i);
    z = element_positions(3, i);

    % Direction vector from element to focal point
    direction = focal_point - [x, y, z];
    direction = direction / norm(direction); % Normalizing the vector

    % Plot element position
    scatter3(x, y, z, 'filled');

    % Plot direction vector
    quiver3(x, y, z, direction(1), direction(2), direction(3), 0.02, 'MaxHeadSize', 2);
end

xlabel('X Position (m)');
ylabel('Y Position (m)');
zlabel('Z Position (m)');
title('Transducer Elements and Orientation Vectors');
axis equal;
grid on;



% =========================================================================
% SETUP GRID
% =========================================================================

fprintf('\n\nSetup Grid.......\n')

% Validate grid size parameters 
% Ensure the grid dimensions are at least twice the maximum radius of the transducer array circles.
if grid_size_x < max(radii) * 2 || grid_size_y < max(radii) * 2 || grid_size_z < max(radii) * 2
    error('Grid size is too small to fit the transducer array.');
end

% Compute the maximum element positions to adjust grid size accordingly
max_x_position = max(element_positions(1, :));
max_y_position = max(element_positions(2, :));
max_z_position = max(element_positions(3, :));

% Adjust grid size based on element positions and add a buffer
Nx = roundEven(max(grid_size_x, max_x_position + 10e-3) / dx);
Ny = roundEven(max(grid_size_y, max_y_position + 10e-3) / dy);
Nz = roundEven(max(grid_size_z, max_z_position + 10e-3) / dz);

% Check if the calculated grid dimensions are smaller than required
if Nx * dx < grid_size_x || Ny * dy < grid_size_y || Nz * dz < grid_size_z
    error('Computed grid dimensions are smaller than required.');
end

% Create the k-Wave grid with adjusted spacing
kgrid = kWaveGrid(Nx, dx, Ny, dx, Nz, dx);

% Set up the simulation time based on the source frequency and cycles
tone_burst_duration = source_cycles / source_f0;  % Duration of the tone burst
t_end = max(t_end, tone_burst_duration);  % Ensure t_end is long enough
kgrid.makeTime(c0, ppw, t_end);  % Calculate kgrid.dt and number of time steps

% % Recalculate the tone burst
% tone_burst = 1e12 * toneBurst(1 / kgrid.dt, source_f0, source_cycles);
% 
% % Debugging: Check the new maximum amplitude and plot
% disp(['Adjusted max amplitude of initial tone burst: ', num2str(max(abs(tone_burst)))]);
% figure;
% plot(tone_burst);
% title('Adjusted Initial Tone Burst');
% 
% 
% % Debugging outputs
% disp(['Adjusted Time Step (kgrid.dt): ', num2str(kgrid.dt)]);
% disp(['Number of time steps (kgrid.Nt): ', num2str(kgrid.Nt)]);
% disp(['Size of tone_burst: ', num2str(length(tone_burst))]);


% =========================================================================
% VISUALIZATION OF GRID AND TRANSDUCER ELEMENTS
% =========================================================================

% Calculate the physical size of the grid for visualization
grid_size_physical_x = Nx * dx;
grid_size_physical_y = Ny * dx;
grid_size_physical_z = Nz * dx;

% % Combined plot showing the grid boundaries and transducer elements
% figure;
% hold on;

% Plotting the grid boundaries
plot3([0, grid_size_physical_x], [0, 0], [0, 0], 'r'); % X-axis
plot3([0, 0], [0, grid_size_physical_y], [0, 0], 'g'); % Y-axis
plot3([0, 0], [0, 0], [0, grid_size_physical_z], 'b'); % Z-axis

% Plotting the transducer elements
% scatter3(element_positions(1,:), element_positions(2,:), element_positions(3,:), 'filled');
% 
% % Setting labels and title for the plot
% xlabel('X-axis (m)');
% ylabel('Y-axis (m)');
% zlabel('Z-axis (m)');
% title('Transducer Elements within Computational Grid');
% axis equal;
% grid on;



% =========================================================================
% CALCULATE TIME DELAYS
% =========================================================================

fprintf('\n\nCalculate Time Delays....\n')

% Calculate time delays
time_delays = zeros(1, total_elements); % Initialize the time delays array
 % zero array is initialized to store the time delay for each transducer element
% note that beam steering is built off time delays and is fundamental to
% phased arrays

element_positions = karray.getElementPositions(); % 3d positions of all the elements in the transducer array
for ind = 1:total_elements
    distance = norm(element_positions(:, ind) - focal_point'); % Distance from element to focal point
    time_delays(ind) = distance / c0;  % Time delay based on distance and speed of sound
end

% Normalize time delays
time_delays = time_delays - min(time_delays);

% Visualize Time Delays
figure;
scatter3(element_positions(1,:), element_positions(2,:), element_positions(3,:), 36, time_delays, 'filled');
title('Time Delays of Elements');
xlabel('X Position (m)');
ylabel('Y Position (m)');
zlabel('Z Position (m)');
colorbar;
grid on;
axis equal;

% =========================================================================
% CREATE CUSTOM SOURCE MASK
% =========================================================================

% Creating the Custom Source Mask
fprintf("\n\nCreate Custom Source Mask....\n")
source_mask = zeros(Nx, Ny, Nz);
for ind = 1:total_elements
    % Convert positions to grid indices
    grid_index_x = round((element_positions(1, ind) + grid_size_x / 2) / dx) + 1;
    grid_index_y = round((element_positions(2, ind) + grid_size_y / 2) / dy) + 1;
    grid_index_z = round((element_positions(3, ind) + grid_size_z / 2) / dz) + 1;

    % Validate indices
    if grid_index_x < 1 || grid_index_x > Nx || grid_index_y < 1 || grid_index_y > Ny || grid_index_z < 1 || grid_index_z > Nz
        error('Source element %d is out of grid bounds. Indices: X=%d, Y=%d, Z=%d', ind, grid_index_x, grid_index_y, grid_index_z);
    end

    % Set the source mask
    source_mask(grid_index_x, grid_index_y, grid_index_z) = 1;
end
source.p_mask = source_mask;

% Verify the number of elements
if nnz(source.p_mask) ~= total_elements
    error('Mismatch in source mask elements. Expected: %d, Found: %d', total_elements, nnz(source.p_mask));
end
fprintf('Source mask created with %d elements.\n', nnz(source.p_mask));


% Print non-zero elements of the source mask for verification
fprintf('Non-zero elements in source mask: %d\n', nnz(source.p_mask));


% =========================================================================
% CREATE SOURCE SIGNALS WITH TIME DELAYS
% =========================================================================

fprintf('\nCreating Source Signals with Time Delays...\n');

% Increase the amplitude of the tone burst
amplitude_factor = 2;  % Adjust this factor as needed
tone_burst = amplitude_factor * 1e6 * toneBurst(1 / kgrid.dt, source_f0, source_cycles);
disp(['Max amplitude of initial tone burst: ', num2str(max(abs(tone_burst)))]);

% Initialize source_sig matrix to store individual signals for each element
source_sig = zeros(kgrid.Nt, total_elements);
disp(['Initialized source_sig matrix size: ', num2str(size(source_sig))]);

% Iterate over each transducer element
for ind = 1:total_elements
    fprintf('Processing element %d of %d\n', ind, total_elements);

    % Calculate the delay for this element in samples
    delay_samples = round(time_delays(ind) / kgrid.dt);
    disp(['Element ', num2str(ind), ' delay in samples: ', num2str(delay_samples)]);

    % Calculate start and end indices for placing the tone burst
    start_idx = max(1, delay_samples + 1);
    end_idx = min(start_idx + length(tone_burst) - 1, kgrid.Nt);

    % Check for overlap or out-of-bounds issues
    if start_idx > kgrid.Nt
        disp(['Skipping element ', num2str(ind), ' (start index exceeds grid size)']);
        continue;
    elseif end_idx > kgrid.Nt
        disp(['Truncating signal for element ', num2str(ind)]);
        end_idx = kgrid.Nt;
    end

    % Adjust tone burst length if necessary and place it in source_sig
    adjusted_tone_burst = tone_burst(1:(end_idx - start_idx + 1));
    source_sig(start_idx:end_idx, ind) = adjusted_tone_burst;

    % Debugging: Check if the tone burst is placed correctly
    if any(source_sig(:, ind) > 0)
        disp(['Tone burst placed for element ', num2str(ind)]);
    else
        disp(['Warning: No tone burst placed for element ', num2str(ind)]);
    end
end

% Combine all individual signals into one
source.p = sum(source_sig, 2);

% Reshape source.p to match the dimensions of source.p_mask
non_zero_indices = find(source.p_mask);
reshaped_source_p = zeros(size(source.p_mask));
reshaped_source_p(non_zero_indices) = source.p(1:length(non_zero_indices));

% Replace the original source.p with the reshaped version
source.p = reshaped_source_p;

% Debugging: Check if the dimensions of source.p are consistent with source.p_mask
if numel(source.p) ~= numel(source.p_mask)
    error('Mismatch in dimensions of source.p and source.p_mask.');
else
    disp('source.p dimensions are consistent with source.p_mask.');
end

% Optionally, plot the reshaped combined signal for visual inspection
figure;
imagesc(max(reshaped_source_p, [], 3)); % Show the max projection along one axis
title('Reshaped Combined Source Signal');
xlabel('X Position');
ylabel('Y Position');
colorbar;
