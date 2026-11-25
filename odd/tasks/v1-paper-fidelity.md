# v1-paper-fidelity

Objective: make the v1 code (src/RPINN + RPINN/Notebooks) reproduce the published RPINN paper (Computers 2024, 13, 176) and regenerate every v1 model, result, figure and table from it. Branch `fix/v1-paper-fidelity` (from `feat/v2-sources-unsupplied`). Delivery: ask-on-risk. Test runner: `.venv/bin/python -m pytest tests`.

## Specs

- S1 Scope is v1 only, the 2024 paper: "ojo paper 2024 no sobre lo nueevo". Nothing in src/CLRPINN changes.
- S2 The code follows the paper: "revisar que el v1 este acorde al paper y correr tanto IPOPT y RPINNN todo bajo el esquema del papeer nada  más". The paper governs every item of the audit except the compression-ratio penalty equation (code kept, it matches the paper's prose "middle of range"; the equation is the one to correct). Approved: "si, no hay lío".
- S3 Gas loss carries no cost term: "en redes no colocaría el costo donde los otras restricciones den, es la que menor costo tiene, es que al no estar normalizado en comparación esos loss se ven opacados respecto al costo, por eso se lo quite". λ_L = 0 for the gas network; cost is measured at evaluation only. The paper text must state λ_L = 0 (paper note, not code).
- S4 Gas architecture matches Table tab:arch-gas: 12,707 parameters on the 8-node network (all weights, trainable and non-trainable, fixed balance weights included). Layers: Dense_1 236 SELU from the input; one shared BatchNormalization_1; partial flows (pipes, compressors from BN_1; other classes and storage from the input, no BN); Dense_2 N SELU + BatchNormalization_2 feeding the unsupplied-gas switch; source switch with 1 weight; pressure branch Dense_3 236 SELU from the input + BatchNormalization_3 + bounded pressure layer that reads BN_3's output.
- S5 Gas training per paper: 20,000 samples per network and scenario, 30% test, seeded; 600 epochs, batch 256, Adamax lr 1e-2 with decreasing schedule (x0.1 at epochs 400, 500, 580); λ_Bal, λ_Wey, λ_ratio = 0.6, 0.1, 0.3 (the 3-output scheme of the saved weights); scenarios S/M/D by the `ajus` offsets of the training notebook; no retrain-until-good loop. A training entry point lives in src.
- S6 IPOPT per paper: GEKKO local IPOPT, MPCC Weymouth, on the same 320 test samples per network and scenario as RPINN; 60 s cap per sample (declared: not in the paper); non-converged samples reported separately, never dropped silently. Approved: "si, no hay lío".
- S7 Mixture model per Table tab:arch-mix: Dense_1 5 SELU (25) + Dense_2 1 SELU with l1/max constraint (5) = 30 parameters; loss λ·Huber + λ·sum-to-one penalty; Adam lr 1e-3, 600 epochs, batch 32; 500 samples (400 train with 30% validation, 100 test), 100 repetitions, SNR {-1, 3, 5, None}, seeded; comparator IPOPT with an l2 objective.
- S8 Metrics per paper: MAPE x100 in %, reference value as truth; one compression-violation measure for both methods; cost panel shows the cost difference RPINN - IPOPT with one cost vector.
- S9 Regenerate everything: "regenerear todo es todo, modelos, results, figuras tablas etc. actulizar lo notrebooks apra que tomen estos nuevos datos, que si se sobrescriben solo es garantizar que las neuvas arquitecturas se acoplen". Figures 7-12 and the architecture tables; notebooks read the new files.
- S10 Run on the fastest local machine, measured with and without GPU, Mac GPU included: "hacerlo sobre la maquina local que mas rapido lo haga etimando timepos con o sin gpu incluida la maac con su propia gpu y cuanto timmpo se demoraría".

## Tasks

| ID | Specs | Route | Status | Evidence |
|---|---|---|---|---|
| T0 | S10 | inline | done | Mac M2 CPU 0.016-0.030 s/epoch (current arch, ~900 samples); Mac Metal GPU 0.205-0.239 s/epoch; IPOPT 0.05-0.08 s solve when it converges, hangs without a cap |
| T1 | S3 S4 | inline | done | RED 3 failed (127467 params, 2 outputs) -> GREEN 4 passed; test_portability old-weight load fails until T6 regenerates the weights |
| T2 | S5 | inline | pending | |
| T3 | S6 S8 | inline | pending | |
| T4 | S7 S8 | inline | pending | |
| T5 | S10 | inline | pending | timing of the new architecture, then full estimate |
| T6 | S9 | inline | pending | full regeneration run |
| T7 | S9 | inline | pending | notebooks + figures 7-12 + tables |

## Log

- L1 (2026-11-25) "primero ajustar para que se parezca al paper . no te estoy diciendo ni mierda de eso, te estoy diciendo, ese v1 me parece a mi que es mejor revisar que el v1 este acorde al paper y correr tanto IPOPT y RPINNN todo bajo el esquema del papeer nada  más, dudas ???? entnediste bien, ?? ojo paper 2024 no sobre lo nueevo"
- L2 (2026-11-25) "pero regenerear todo es todo, modelos, results, figuras tablas etc. actulizar lo notrebooks apra que tomen estos nuevos datos, que si se sobrescriben solo es garantizar que las neuvas arquitecturas se acoplen y hacerlo sobre la maquina local que mas rapido lo haga etimando timepos con o sin gpu incluida la maac con su propia gpu y cuanto timmpo se demoraría"
- L3 (2026-11-25) Audit (read-only explorer): saved Modelo2 weights have 127,467 params and no code produces them; gas loss lacks cost; mixture is 5 params/relu/Adamax/100 epochs vs SCS l1; IPOPT comparison filtered to converged samples (D: 3-13 per net); MAPE as fraction. Paper table decomposes exactly to 12,707 with the S4 layout (partial flows 1896 + 378 = 2274).
- L4 (2026-11-25) "si, no hay lío, pero en redes no colocaría el costo donde los otras restricciones den, es la que menor costo tiene, es que al no estar normalizado en comparación esos loss se ven opacados respecto al costo, por eso se lo quite"
- L5 (2026-11-25) T1: gas model rebuilt to tab:arch-gas (12,707 on net 8), losses split into WeymouthLoss and RatioPenaltyLoss with LOSS_WEIGHTS 0.6/0.1/0.3, no cost term. Known failure until T6: tests/test_portability.py loads the pre-change weights.
- Next: T2.
