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



% Adjusted Grid parameters - ensuring the grid is sufficiently large
% grid_size_x = max(60e-3, max(radii) * 2);  % grid size in x [m]
% grid_size_y = max(60e-3, max(radii) * 2);  % grid size in y [m]
% grid_size_z = max(40e-3, max(radii) * 2);  % grid size in z [m]

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


% FOR FLAT:
    % % Add elements to each circle with 3D positioning
    % for circle = 1:length(radii)
    %     radius = radii(circle);
    %     num_elements = elements_per_circle(circle);
    % 
    %     % Set a constant z_pos for each circle
    %     z_pos = circle * some_constant;  % some_constant determines the spacing between circles
    % 
    %     for ind = 1:num_elements
    %         angle = 2 * pi * (ind - 1) / num_elements;
    %         x_pos = radius * cos(angle);
    %         y_pos = radius * sin(angle);
    % 
    %         % Calculate the direction vector from element to the focal point
    %         direction = focal_point - [x_pos, y_pos, z_pos];
    % 
    %         % Calculate pitch (rotation around x-axis) and yaw (rotation around y-axis)
    %         pitch = atan2(-direction(3), sqrt(direction(1)^2 + direction(2)^2)); % negative z to point upwards
    %         yaw = atan2(direction(2), direction(1));
    % 
    %         % Assuming roll (rotation around z-axis) is zero
    %         roll = 0;
    % 
    %         theta = [pitch, yaw, roll];
    %         karray.addRectElement([x_pos, y_pos, z_pos], element_width, element_length, theta);
    % 
    %     end
    % end


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
% VERIFY ELEMENT POSITIONS AND PRINT RANGE
% =========================================================================

% % Get positions of all elements
% element_positions = karray.getElementPositions();
% 
% % Check if any element is outside the grid boundaries
% if any(element_positions(1, :) > grid_size_x) || ...
%    any(element_positions(2, :) > grid_size_y) || ...
%    any(element_positions(3, :) > grid_size_z)
%     error('Some elements are outside the grid boundaries.');
% end
% 
% % Print out the range of element positions
% fprintf('Range of x positions: [%f, %f] m\n', min(element_positions(1, :)), max(element_positions(1, :)));
% fprintf('Range of y positions: [%f, %f] m\n', min(element_positions(2, :)), max(element_positions(2, :)));
% fprintf('Range of z positions: [%f, %f] m\n', min(element_positions(3, :)), max(element_positions(3, :)));
% 
% 
% % Ensure all element positions are within the grid boundaries
% if any(element_positions(1, :) > grid_size_x / 2) || ...
%    any(element_positions(1, :) < -grid_size_x / 2) || ...
%    any(element_positions(2, :) > grid_size_y / 2) || ...
%    any(element_positions(2, :) < -grid_size_y / 2) || ...
%    any(element_positions(3, :) > grid_size_z / 2) || ...
%    any(element_positions(3, :) < -grid_size_z / 2)
%     error('Element positions are outside the grid boundaries.');
% end
% 
% 
% % After calculating element positions
% max_element_x = max(element_positions(1, :));
% min_element_x = min(element_positions(1, :));
% max_element_y = max(element_positions(2, :));
% min_element_y = min(element_positions(2, :));
% max_element_z = max(element_positions(3, :));
% min_element_z = min(element_positions(3, :));
% 
% fprintf('Element position X range: [%f, %f]\n', min_element_x, max_element_x);
% fprintf('Element position Y range: [%f, %f]\n', min_element_y, max_element_y);
% fprintf('Element position Z range: [%f, %f]\n', min_element_z, max_element_z);

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

% Recalculate the tone burst
tone_burst = 1e6 * toneBurst(1 / kgrid.dt, source_f0, source_cycles);

% Debugging outputs
disp(['Adjusted Time Step (kgrid.dt): ', num2str(kgrid.dt)]);
disp(['Number of time steps (kgrid.Nt): ', num2str(kgrid.Nt)]);
disp(['Size of tone_burst: ', num2str(length(tone_burst))]);




% =========================================================================
% VISUALIZATION OF GRID AND TRANSDUCER ELEMENTS
% =========================================================================

% Calculate the physical size of the grid for visualization
grid_size_physical_x = Nx * dx;
grid_size_physical_y = Ny * dx;
grid_size_physical_z = Nz * dx;

% Combined plot showing the grid boundaries and transducer elements
figure;
hold on;

% Plotting the grid boundaries
plot3([0, grid_size_physical_x], [0, 0], [0, 0], 'r'); % X-axis
plot3([0, 0], [0, grid_size_physical_y], [0, 0], 'g'); % Y-axis
plot3([0, 0], [0, 0], [0, grid_size_physical_z], 'b'); % Z-axis

% Plotting the transducer elements
scatter3(element_positions(1,:), element_positions(2,:), element_positions(3,:), 'filled');

% Setting labels and title for the plot
xlabel('X-axis (m)');
ylabel('Y-axis (m)');
zlabel('Z-axis (m)');
title('Transducer Elements within Computational Grid');
axis equal;
grid on;



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

% % Checkpoint: Check the time delays to ensure its working perfectly:
% fprintf('Total number of elements: %d\n', total_elements);
% fprintf('Focal point coordinates: [%f, %f, %f] m\n', focal_point);
% fprintf('Time delay for first element: %f microseconds\n', time_delays(1) * 1e6);
% fprintf('Time delay for middle element: %f microseconds\n', time_delays(round(total_elements/2)) * 1e6);
% fprintf('Time delay for last element: %f microseconds\n', time_delays(end) * 1e6);
% fprintf('Minimum time delay: %f microseconds\n', min(time_delays) * 1e6);
% fprintf('Maximum time delay: %f microseconds\n', max(time_delays) * 1e6);
% [min_delay, min_idx] = min(time_delays);
% [max_delay, max_idx] = max(time_delays);
% fprintf('Closest element to focal point (Element %d): Distance = %f m, Time Delay = %f microseconds\n', min_idx, norm(element_positions(:, min_idx) - focal_point'), min_delay * 1e6);
% fprintf('Farthest element from focal point (Element %d): Distance = %f m, Time Delay = %f microseconds\n', max_idx, norm(element_positions(:, max_idx) - focal_point'), max_delay * 1e6);
% for ind = 1:total_elements
%     fprintf('Time delay for element %d: %f microseconds\n', ind, time_delays(ind) * 1e6);
% end

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
% VISUALIZE SOURCE MASK WITH 3D SCATTER PLOT
% =========================================================================

% Visualization of the Source Mask
[x_source, y_source, z_source] = ind2sub(size(source_mask), find(source_mask));

figure;
scatter3(x_source * dx, y_source * dy, z_source * dz, 'filled');
xlabel('X Position (m)');
ylabel('Y Position (m)');
zlabel('Z Position (m)');
title('Visualization of Source Mask in Grid');
axis equal;
grid on;
% 
% % Initialize arrays to store grid indices
% grid_indices_x = zeros(1, total_elements);
% grid_indices_y = zeros(1, total_elements);
% grid_indices_z = zeros(1, total_elements);
% 
% % Calculate grid indices and set source mask
% for ind = 1:total_elements
%     grid_indices_x(ind) = round((element_positions(1, ind) + grid_size_x / 2) / dx) + 1;
%     grid_indices_y(ind) = round((element_positions(2, ind) + grid_size_y / 2) / dy) + 1;
%     grid_indices_z(ind) = round((element_positions(3, ind) + grid_size_z / 2) / dz) + 1;
% 
%     fprintf('Element %d: Grid Position (X, Y, Z) = (%d, %d, %d)\n', ind, grid_indices_x(ind), grid_indices_y(ind), grid_indices_z(ind));
% 
%     % Check if the element is within the grid bounds
%     if grid_indices_x(ind) < 1 || grid_indices_x(ind) > Nx || ...
%        grid_indices_y(ind) < 1 || grid_indices_y(ind) > Ny || ...
%        grid_indices_z(ind) < 1 || grid_indices_z(ind) > Nz
%         fprintf('Element %d is out of grid bounds at position (X, Y, Z) = (%d, %d, %d)\n', ind, grid_indices_x(ind), grid_indices_y(ind), grid_indices_z(ind));
%     else
%         % Set the source mask
%         source_mask(grid_indices_x(ind), grid_indices_y(ind), grid_indices_z(ind)) = 1;
%     end
% end
% 
% % Verify source mask settings
% for ind = 1:total_elements
%     if source_mask(grid_indices_x(ind), grid_indices_y(ind), grid_indices_z(ind)) == 1
%         fprintf('Source mask correctly set for element %d at position (X, Y, Z) = (%d, %d, %d)\n', ind, grid_indices_x(ind), grid_indices_y(ind), grid_indices_z(ind));
%     else
%         fprintf('Mismatch in source mask for element %d at position (X, Y, Z) = (%d, %d, %d)\n', ind, grid_indices_x(ind), grid_indices_y(ind), grid_indices_z(ind));
%     end
% end
% 
% % Confirm the number of non-zero elements in the source mask
% num_non_zero_elements_in_mask = nnz(source_mask);
% fprintf('Number of non-zero elements in source mask: %d\n', num_non_zero_elements_in_mask);
% if num_non_zero_elements_in_mask == total_elements
%     fprintf('Number of elements in source mask matches the total number of elements.\n');
% else
%     fprintf('Mismatch in the number of elements. Expected: %d, Found in mask: %d\n', total_elements, num_non_zero_elements_in_mask);
% end

% =========================================================================
% CREATE SOURCE SIGNALS WITH TIME DELAYS
% =========================================================================

fprintf('\n\n--- Creating Source Signals with Time Delays ---\n');

% Display basic grid and tone burst information
disp(['Grid time step (kgrid.dt): ', num2str(kgrid.dt), ' seconds']);
disp(['Number of time steps (kgrid.Nt): ', num2str(kgrid.Nt)]);
disp(['Length of tone_burst array: ', num2str(length(tone_burst)), ' samples']);

% Initialize source_sig with zeros to store individual signals
source_sig = zeros(kgrid.Nt, total_elements, 'single');
disp(['Initialized source_sig matrix size: ', num2str(size(source_sig))]);

% Iterate over each transducer element
for ind = 1:total_elements
    fprintf('\nProcessing element %d of %d\n', ind, total_elements);

    % Calculate the delay for this element
    delay_samples = round(time_delays(ind) / kgrid.dt);
    disp(['Element ', num2str(ind), ' delay: ', num2str(time_delays(ind)), ' seconds (', num2str(delay_samples), ' samples)']);

    % Calculate the indices in source_sig where the tone burst will be placed
    start_idx = max(1, delay_samples + 1);
    end_idx = min(kgrid.Nt, start_idx + length(tone_burst) - 1);
    disp(['Start index: ', num2str(start_idx), ', End index: ', num2str(end_idx)]);

    % Adjust the tone_burst length if necessary
    adjusted_tone_burst_length = end_idx - start_idx + 1;
    disp(['Adjusted tone burst length: ', num2str(adjusted_tone_burst_length), ' samples']);

    if adjusted_tone_burst_length <= length(tone_burst)
        adjusted_tone_burst = tone_burst(1:adjusted_tone_burst_length);
    else
        adjusted_tone_burst = [tone_burst, zeros(1, adjusted_tone_burst_length - length(tone_burst))];
    end

    % Display information about the adjusted tone burst
    disp(['Length of adjusted tone burst for element ', num2str(ind), ': ', num2str(length(adjusted_tone_burst))]);

    % Assign the adjusted tone burst to the appropriate column in source_sig
    source_sig(start_idx:end_idx, ind) = adjusted_tone_burst(:);
end

% Transpose source_sig to have elements along the second dimension
source_sig = source_sig.';

% Sum all individual signals into one combined signal for source.p
source.p = sum(source_sig, 1);
disp(['Size of combined source signal (source.p): ', num2str(size(source.p))]);

% Debugging: Check the max and min values in the combined signal
disp(['Max value in source.p: ', num2str(max(source.p))]);
disp(['Min value in source.p: ', num2str(min(source.p))]);

% Optionally, plot the combined signal for visual inspection
figure;
plot(source.p);
title('Combined Source Signal');
xlabel('Time Step');
ylabel('Amplitude');

% Reshape source.p to match the dimensions of source.p_mask
source.p = reshapeSourceP(source.p, source.p_mask);

% Confirm the size of source.p matches the number of non-zero elements in source.p_mask
if length(source.p) ~= nnz(source.p_mask)
    error('Mismatch in dimensions of source.p and source.p_mask.');
end

disp('source.p reshaped to match source.p_mask.');

% Optionally, plot the combined signal for visual inspection
figure;
plot(source.p);
title('Combined Source Signal');
xlabel('Time Step');
ylabel('Amplitude');

% Visualize the source mask
figure;
imagesc(max(source.p_mask, [], 3)); % Show the max projection along one axis
title('Source Mask Visualization');
xlabel('X Position');
ylabel('Y Position');
colorbar;



% =========================================================================
% SETUP MEDIUM AND SENSOR
% =========================================================================

fprintf('\n\nSetting up Medium and Sensor....\n');

% Define medium properties
medium.sound_speed = c0;  % Sound speed [m/s]
medium.density = rho0;    % Density [kg/m^3]

% % Define a focused sensor mask
% sensor_radius = max(radii) + 10e-3; % Extend beyond the largest radius of the transducer
% sensor_height = 20e-3; % Height of the sensor plane above the transducer
% sensor.mask = zeros(Nx, Ny, Nz);
% [x, y, z] = meshgrid(1:Nx, 1:Ny, 1:Nz);
% x_center = Nx/2;
% y_center = Ny/2;
% z_transducer = round(max(element_positions(3, :)) / dz);
% sensor.mask(((x - x_center).^2 + (y - y_center).^2 <= (sensor_radius/dx)^2) & (z >= z_transducer) & (z <= z_transducer + sensor_height/dz)) = 1;
% 
% % Specify what to record
% sensor.record = {'p_max', 'p_final'};
% 
% % Visualize the sensor mask
% figure;
% voxelPlot(double(sensor.mask));
% title('Sensor Mask Visualization');
% xlabel('X');
% ylabel('Y');
% zlabel('Z');



% Define a focused sensor mask
sensor_radius = max(radii) + 10e-3;  % Extend beyond the largest radius of the transducer
sensor_height = 20e-3;  % Height of the sensor plane above the transducer

% Calculate the depth of the transducer
z_transducer_depth = round(max(element_positions(3, :)) / dz);

% Adjust sensor mask to cover the region from the transducer to the focal point
z_sensor_start = z_transducer_depth;  % Start of the sensor mask
z_sensor_end = z_sensor_start + round(sensor_height / dz);  % End of the sensor mask

% Create the sensor mask
sensor.mask = zeros(Nx, Ny, Nz);
[x, y, z] = meshgrid(1:Nx, 1:Ny, 1:Nz);
x_center = Nx / 2;
y_center = Ny / 2;
sensor.mask(((x - x_center).^2 + (y - y_center).^2 <= (sensor_radius / dx)^2) & (z >= z_sensor_start) & (z <= z_sensor_end)) = 1;

% Visualize the sensor mask
figure;
voxelPlot(double(sensor.mask));
title('Sensor Mask Visualization');
xlabel('X');
ylabel('Y');
zlabel('Z');





% =========================================================================
% RUN SIMULATION
% =========================================================================

fprintf('\nRunning Simulation...\n');

% Set input options
input_args = {...
    'PMLSize', 'auto', ...
    'PMLInside', false, ...
    'PlotPML', false, ...
    'DisplayMask', 'off', ...
    'DataCast', 'single', ...
    'PlotScale', [-1, 1] * 1e6};

% Run the simulation
sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor, input_args{:});

fprintf('Simulation Complete.\n');

% =========================================================================
% POST-PROCESSING AND VISUALIZATION
% =========================================================================

% Convert the recorded data to absolute values
abs_pressure = abs(sensor_data);

% Use the maximum absolute pressure value for scaling
max_pressure = max(abs_pressure(:));

% Normalize the pressure data
normalized_pressure = abs_pressure / max_pressure;

% Avoid log of zero by adding a small number
normalized_pressure(normalized_pressure == 0) = 1e-10;

% Calculate the logarithmic scale
log_pressure = 20 * log10(normalized_pressure);

% Plot the logarithmic scale results
figure;
imagesc(log_pressure);
colormap('hot'); % Enhance visibility with colormap
colorbar;
title('Logarithmic Scale of Normalized Pressure Field');
xlabel('Sensor Element');
ylabel('Time Step');
axis image; % Correct axis scaling

% Debugging: Plot raw pressure data
figure;
imagesc(abs_pressure);
colormap('jet'); % Use a different colormap for raw data
colorbar;
title('Raw Pressure Data');
xlabel('Sensor Element');
ylabel('Time Step');
axis image; % Correct axis scaling

% =========================================================================
% VERIFYING SOURCE SIGNAL
% =========================================================================

% Check if source.p contains mostly non-zero values
if nnz(source.p) / numel(source.p) < 0.5
    warning('More than half of the source.p values are zero. Check the source signal generation.');
end

% Display some statistics about source.p
disp(['Max value in source.p: ', num2str(max(source.p))]);
disp(['Min value in source.p: ', num2str(min(source.p))]);
disp(['Mean value in source.p: ', num2str(mean(source.p))]);

% =========================================================================
% SENSOR CONFIGURATION VERIFICATION
% =========================================================================

% Check if the sensor mask covers the expected region
expected_sensor_coverage = pi * sensor_radius^2 * sensor_height;
actual_sensor_coverage = nnz(sensor.mask) * dx * dy * dz;

if abs(expected_sensor_coverage - actual_sensor_coverage) / expected_sensor_coverage > 0.1
    warning('The sensor coverage area differs significantly from the expected value. Check the sensor configuration.');
end

% =========================================================================
% ADDITIONAL VISUALIZATIONS (OPTIONAL)
% =========================================================================

% Example of a 3D plot (consider only for a smaller subset of data)
% [x_grid, y_grid, z_grid] = meshgrid(1:Nx, 1:Ny, 1:Nz);
% scatter3(x_grid(:), y_grid(:), z_grid(:), 10, abs_pressure(:), 'filled');
% xlabel('X');
% ylabel('Y');
% zlabel('Z');
% title('3D Pressure Field');






function reshaped_p = reshapeSourceP(source_p, source_p_mask)
    % Ensure source_p is a row vector
    if size(source_p, 1) > 1
        source_p = source_p.';
    end

    % Initialize the reshaped source signal with zeros
    reshaped_p = zeros(sum(source_p_mask(:)), 1);
    
    % Find the indices of the non-zero elements in the source mask
    non_zero_indices = find(source_p_mask);

    % Loop over each non-zero element and assign the corresponding time series
    for i = 1:length(non_zero_indices)
        reshaped_p(i) = source_p(i);
    end
end
