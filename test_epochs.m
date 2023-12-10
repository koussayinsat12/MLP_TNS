L = 5000;
n = 1:L;
phi = 1.5 * rand(1, L);
wo = 10 * pi;
d = sin(n * wo + phi)';
h = [-1.5; 0.5; 0.2];
D = zeros(length(h), 1);
x = zeros(L, 1);
for i = 1:L
    D = [d(i); D(1:length(h) - 1)];
    x(i) = h' * D + (h' * D)^3;
end
N_train = 2000;
X_train = zeros(N_train, 1);
d_train = zeros(N_train, 1);
X_train(:) = x(1:N_train);
d_train(:) = d(1:N_train);
X_test = x(N_train + 1:end);
d_test = d(N_train + 1:end);
params.n_neurons = 30;
params.order = 5;
params.step = 0.01;
params.alpha_init = 'randn';
params.w_init = 'randn';
params.batch_size = N_train;
epochs_values = [10, 20, 30, 40, 50];
eqm_values = cell(1, length(epochs_values));
Wopt_matrices = cell(1, length(epochs_values));
figure;
hold on;
for idx = 1:length(epochs_values)
    epochs = epochs_values(idx);
    mlp = MLP(params);
    [eqm, trained_model] = mlp.train(epochs, X_train, d_train, 'constant');
    eqmin = mean(eqm(calcul_eqmmin(eqm, 0.0001):end));
    eqm_values{idx} = eqm;
    Wopt = trained_model.Wopt;
    Wopt_matrices{idx} = Wopt;
    fprintf('Pour epochs = %d, eqmmin = %f\n', epochs, eqmin);
end
for idx = 1:length(epochs_values)
    plot(eqm_values{idx}, 'LineWidth', 1.5);
end
legend('Époques 10', 'Époques 20', 'Époques 30', 'Époques 40', 'Époques 50');
xlabel('Époques');
ylabel('EQM');
title('EQM pour différentes valeurs d epoques');
hold off;
for idx = 1:length(epochs_values)
    Wopt = Wopt_matrices{idx};
    figure;
    imagesc(Wopt);
    colormap('jet');
    colorbar;
    title(sprintf('Matrice de poids entraînée (Époques %d)', epochs_values(idx)));
    xlabel('Indice du neurone');
    ylabel('Indice de l entree');
end

