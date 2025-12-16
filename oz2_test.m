function oz2_test

gd = gpuDevice();
A = gpuArray(randn(1000));
B = gpuArray(randn(1000));

fprintf('[DGEMM]\n');
C = A*B; wait(gd);
for num_moduli=10:15
    D=oz2(A,B,num_moduli,1);
    wait(gd);
    err = (gather(C)-gather(D))./gather(C);
    err_max = max(abs(err),[],'all');
    fprintf('fast: No. moduli = %2d, err = %.1e\n',num_moduli,err_max);
end
for num_moduli=10:15
    D=oz2(A,B,num_moduli,0);
    wait(gd);
    err = (gather(C)-gather(D))./gather(C);
    err_max = max(abs(err),[],'all');
    fprintf('accu: No. moduli = %2d, err = %.1e\n',num_moduli,err_max);
end

fprintf('\n[SGEMM]\n');
C = double(single(A))*double(single(B)); wait(gd);
for num_moduli=4:10
    D=oz2(single(A),single(B),num_moduli,1);
    wait(gd);
    err = (gather(C)-gather(D))./gather(C);
    err_max = max(abs(err),[],'all');
    fprintf('fast: No. moduli = %2d, err = %.1e\n',num_moduli,err_max);
end
for num_moduli=4:10
    D=oz2(single(A),single(B),num_moduli,0);
    wait(gd);
    err = (gather(C)-gather(D))./gather(C);
    err_max = max(abs(err),[],'all');
    fprintf('accu: No. moduli = %2d, err = %.1e\n',num_moduli,err_max);
end

end