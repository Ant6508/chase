"""Test de fumée : le code d'entraînement de RecursiveMAS tourne-t-il avec l'architecture
de google/gemma-4-12B-it ? Voir results/jalon3_recursivemas_gemma4.md.

Modèle miniature à poids aléatoires, sur CPU : même classe et même config que le vrai
(48 couches ramenées à 6, largeur 3840 ramenée à 64), vrai tokenizer et vrai gabarit de
chat. Le test ne dit rien de la qualité ; il dit si le chemin de code passe :
chargement par AutoModelForCausalLM, états cachés, inputs_embeds, gradient jusqu'aux
liens, génération latente avec cache.

Environnement séparé de celui du projet (torch, transformers >= 5.10, datasets) :

    git clone https://github.com/RecursiveMAS/RecursiveMAS <repo>
    # dans <hf> : config.json, tokenizer.json, tokenizer_config.json, chat_template.jinja
    # de https://huggingface.co/google/gemma-4-12B-it (quelques dizaines de Mo, pas les poids)
    python scripts/recursivemas_smoke.py --repo <repo> --hf <hf> --work <dossier> [--patch-template]

Sans --patch-template, la boucle externe sort une perte `nan` : le gabarit de Gemma 4
ajoute au prompt de génération un bloc de pensée vide absent du rendu avec réponse.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="clone de RecursiveMAS")
    parser.add_argument("--hf", required=True, help="config.json et tokenizer de gemma-4-12B-it")
    parser.add_argument("--work", required=True, help="dossier où écrire le modèle miniature")
    parser.add_argument("--patch-template", action="store_true",
                        help="texte complet = prompt + réponse + fin de tour")
    args = parser.parse_args()
    sys.path[:0] = [os.path.join(args.repo, "train"), os.path.join(args.repo, "train", "outer"),
                    args.repo]

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tiny, h = os.path.join(args.work, "g4_tiny"), 64

    # 1. config du vrai 12B, rétrécie
    with open(os.path.join(args.hf, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    t = cfg["text_config"]
    t.update(hidden_size=h, intermediate_size=2 * h, num_hidden_layers=6,
             layer_types=t["layer_types"][:6], num_attention_heads=2, num_key_value_heads=1,
             head_dim=32, global_head_dim=64)
    cfg["vision_config"].update(mm_embed_dim=h, output_proj_dims=h)
    cfg["audio_config"].update(hidden_size=32, audio_embed_dim=32, output_proj_dims=h)
    shutil.rmtree(tiny, ignore_errors=True)
    os.makedirs(tiny)
    with open(os.path.join(tiny, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(tiny))
    model.save_pretrained(tiny)
    for name in os.listdir(args.hf):
        if name != "config.json":
            shutil.copy(os.path.join(args.hf, name), tiny)
    print(f"[1] AutoModelForCausalLM charge {type(model).__name__} ({model.config.model_type})")

    # 2. boucle interne : LatentReasoningModel (train/model.py)
    from model import CrossModelAdapter, LatentReasoningModel

    tok = AutoTokenizer.from_pretrained(tiny)
    msg = [{"role": "user", "content": "Où est la cible ?"},
           {"role": "assistant", "content": "Au nord, à 3 pas par les couloirs."}]
    ids = tok.apply_chat_template(msg, tokenize=True, return_tensors="pt", return_dict=True)
    inner = LatentReasoningModel(tiny, adapter_type="ln_res_adapter", torch_dtype=torch.float32)
    inner.train()
    out = inner(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"])
    out["loss"].backward()
    g_inner = sum(p.grad.abs().sum().item() for p in inner.adapter.parameters() if p.grad is not None)
    frozen = not any(p.grad is not None for p in inner.model.parameters())
    print(f"[2] boucle interne : perte {out['loss'].item():.4f}, gradient sur le lien interne "
          f"{g_inner:.3e}, modèle gelé : {frozen}")
    assert g_inner > 0 and frozen

    # 3. boucle externe : émetteur -> lien interne -> lien externe -> récepteur (common.py)
    import common as C
    from mas_prompt import PLANNER_SLOT

    if args.patch_template:
        render = C.render_chat_text

        def render_chat_text(tokenizer, user_prompt, assistant_text, enable_thinking):
            prompt = render(tokenizer, user_prompt, None, enable_thinking)
            return prompt if assistant_text is None else prompt + assistant_text + "<turn|>\n"

        C.render_chat_text = render_chat_text

    dev = torch.device("cpu")
    sender = C.load_model_and_tokenizer(tiny, dev, torch.float32, False, "émetteur")[0]
    receiver, r_tok = C.load_model_and_tokenizer(tiny, dev, torch.float32, False, "récepteur",
                                                 gradient_checkpointing=True)
    receiver.train()  # comme outer/sequential.py : checkpointing actif, poids gelés
    link_in = inner.adapter.eval()
    for p in link_in.parameters():
        p.requires_grad = False
    link_out = CrossModelAdapter(h, h, "outer_ln_res_adapter")

    with torch.no_grad():
        s_out = sender(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"],
                       output_hidden_states=True, use_cache=False, return_dict=True)
    latent = C.run_inner_adapter_preserve_input_grad(link_in, s_out.hidden_states[-1][0],
                                                     out_dtype=torch.float32)
    emb = receiver.get_input_embeddings()
    message = C.trim_latent(C.run_outer_adapter(link_out, latent, out_dtype=emb.weight.dtype), 8)
    pack = C.build_stage_with_slot(
        tokenizer=r_tok, embedding_layer=emb,
        user_prompt_with_slot=f"Message du partenaire : {PLANNER_SLOT}\nQuel coup joues-tu ?",
        assistant_text="NORD", slot_text=PLANNER_SLOT, slot_embeds=message, enable_thinking=False,
        device=dev, embed_dtype=emb.weight.dtype, max_length=512)
    r_out = receiver(inputs_embeds=pack.inputs_embeds, attention_mask=pack.attention_mask,
                     output_hidden_states=True, use_cache=False, return_dict=True)
    loss = C.compute_solver_ce_loss(r_out.logits, pack.labels)
    loss.backward()
    g_outer = sum(p.grad.abs().sum().item() for p in link_out.parameters() if p.grad is not None)
    frozen = not any(p.grad is not None for p in receiver.parameters())
    print(f"[3] boucle externe : {message.shape[0]} vecteurs dans le contexte du récepteur, "
          f"{int((pack.labels != -100).sum())} tokens de réponse étiquetés, perte {loss.item():.4f}, "
          f"gradient sur le lien externe {g_outer:.3e}, récepteur gelé : {frozen}")
    assert g_outer > 0 and frozen, "perte nan : relancer avec --patch-template"

    # 4. génération latente avec cache : le vecteur produit redevient l'entrée suivante
    receiver.eval()
    with torch.no_grad():
        o = receiver(inputs_embeds=pack.inputs_embeds, attention_mask=pack.attention_mask,
                     use_cache=True, output_hidden_states=True, return_dict=True)
        n = pack.inputs_embeds.shape[1]
        for _ in range(3):
            nxt = C.run_inner_adapter_preserve_input_grad(link_in, o.hidden_states[-1][:, -1:],
                                                          torch.float32)
            n += 1
            o = receiver(inputs_embeds=nxt, attention_mask=torch.ones(1, n, dtype=torch.long),
                         past_key_values=o.past_key_values, use_cache=True,
                         output_hidden_states=True, return_dict=True)
    print(f"[4] génération latente : 3 pas réinjectés avec {type(o.past_key_values).__name__}")
    print("OK")


if __name__ == "__main__":
    main()
