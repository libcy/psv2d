fid = fopen('ux','rt');
tmin = 0; tmax = 0;
umin = 0; umax = 0;

while ~feof(fid)
    data = textscan(fid, '%f %f', 'HeaderLines', 1, 'Delimiter', '\n');
    plot(data{1}, data{2}, 'b'); hold on;
    
    tmin = min(tmin, min(data{1}));
    tmax = max(tmax, max(data{1}));
    
    umin = min(umin, min(data{2}));
    umax = max(umax, max(data{2}));
end
axis tight;
hold off;

fclose(fid);