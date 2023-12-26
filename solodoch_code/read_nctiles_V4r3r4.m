
function field_tiles_out = read_nctiles_V4r3r4(file_name,field_name,sampnum,zlev,ECCOv)
if strcmp(ECCOv,'V4r4')
    addpath .\ECCOv4r4Andrew; 
    field_tiles_out = read_nctiles_daily_V2(file_name,field_name);
elseif strcmp(ECCOv,'V4r3')
    field_tiles_out = read_nctiles(file_name,field_name,sampnum,zlev); %plug in zlev=1 for surface depth if its a 4d field
else
    error('Unfamiliar ECCO version descriptor');
end

