arg = 30;

fid = fopen(sprintf('snap_%d',arg(1)),'rt');
data = textscan(fid, '%f %f %f', 'Delimiter', '\n');
fclose(fid);

x = data{1};
z = data{2};
u = data{3};

n = size(x);
n = n(1);

nx = 0;
nz = 1;
for i = 2:n
    if nz == 1
        nx = nx + 1;
    end
    if x(i) == x(1)
        nz = nz + 1;
    end
end

xa = zeros(nx, 1);
za = zeros(nz, 1);
ua = zeros(nx, nz);

for i = 1:nx
    xa(i) = x(i);
end
for j = 1:nz 
    za(j) =  z(1 + nx * (j - 1));
end

n = size(arg);
n = n(2);

bottom = [0 0 0.5];
botmiddle = [0 0.5 1];
middle = [1 1 1];
topmiddle = [1 0 0];
top = [0.5 0 0];
m = 256;

for isnap=1:n
    fid = fopen(sprintf('snap_%d',arg(isnap)),'rt');
    data = textscan(fid, '%f %f %f', 'Delimiter', '\n');
    fclose(fid);
    u = data{3};

    for i = 1:nx
        for j = 1:nz
            ua(i, j) = u(i + nx * (j - 1));
        end
    end

    imagesc(xa, za, ua');
    
    lims = get(gca, 'CLim');
    if (lims(1) < 0) && (lims(2) > 0)
        ratio = abs(lims(1)) / (abs(lims(1)) + lims(2));
        neglen = round(m*ratio);
        poslen = m - neglen;
        new = [bottom; botmiddle; middle];
        len = length(new);
        oldsteps = linspace(0, 1, len);
        newsteps = linspace(0, 1, neglen);
        newmap1 = zeros(neglen, 3);
        for i=1:3
            newmap1(:,i) = min(max(interp1(oldsteps, new(:,i), newsteps), 0), 1);
        end
        new = [middle; topmiddle; top];
        len = length(new);
        oldsteps = linspace(0, 1, len);
        newsteps = linspace(0, 1, poslen);
        newmap2 = zeros(poslen, 3);
        for i=1:3
            newmap2(:,i) = min(max(interp1(oldsteps, new(:,i), newsteps), 0), 1);
        end
        newmap = [newmap1; newmap2];
    elseif lims(1) >= 0
        new = [middle; topmiddle; top];
        len = length(new);
        oldsteps = linspace(0, 1, len);
        newsteps = linspace(0, 1, m);
        newmap = zeros(m, 3);
        for i=1:3
            newmap(:,i) = min(max(interp1(oldsteps, new(:,i), newsteps), 0), 1);
        end
    else
        new = [bottom; botmiddle; middle];
        len = length(new);
        oldsteps = linspace(0, 1, len);
        newsteps = linspace(0, 1, m);
        newmap = zeros(m, 3);

        for i=1:3
            newmap(:,i) = min(max(interp1(oldsteps, new(:,i), newsteps), 0), 1);
        end

    end
    colormap(newmap)
    drawnow 
end

