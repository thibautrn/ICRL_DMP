import os
import jinja2
import numpy as np
from pathlib import Path


def load_dmp_weights(episode_id, logs_dir):
    """Load DMP weights — finds any file matching *<episode_id>*weight*.npz"""
    matches = list(Path(logs_dir).glob(f"*{episode_id}*weight*.npz"))
    if not matches:
        return None

    weight_file = matches[0]
    try:
        data = np.load(weight_file)
        return {
            'weights': data['w'],
            'y0':      data['y0'],
            'g':       data['g']
        }
    except Exception as e:
        print(f"Error loading DMP weights for {episode_id}: {e}")
        return None


def build_examples(episodes, logs_dir, n_show=20):
    examples_text = ""
    for i, ep in enumerate(episodes[:n_show]):
        dmp_data = load_dmp_weights(ep['episode_id'], logs_dir)
        if dmp_data is not None:
            weights_flat = dmp_data['weights'].flatten().tolist()
            examples_text += "\n" + "---"*20 + f"Example {i+1}" + "---"*20 + "\n"
            examples_text += f"weights = {weights_flat}\n"
            examples_text += f"reward = {ep['total_reward']:.1f}"

    return examples_text


def build_prompt(baseline, iter, max_iters, episodes, logs_dir, history, n_show=20):
    # cup_center kept in signature for compatibility but not used
    N_BFS         = baseline['M']
    TOTAL_WEIGHTS = N_BFS * 3
    data = {
        "max_iters":     max_iters,
        "TOTAL_WEIGHTS": TOTAL_WEIGHTS,
        "iter":          iter,
        "examples_text": build_examples(episodes, logs_dir, n_show),
        "N_BFS":         N_BFS,
        "history":       history,
    }

    template_dir = os.path.join(os.getcwd(), 'templates')
    if not os.path.isdir(template_dir):
        raise FileNotFoundError(f"templates directory not found at {template_dir}")

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        autoescape=False,
    )

    template = env.get_template('initial_prompt.j2')
    return template.render(**data)


def iteration_history(iteration, weights, reward_data, history):
    history += "\n" + "---"*20 + f"Iteration {iteration}" + "---"*20 + "\n"
    history += f"weights = {weights.flatten()}\n"
    history += f"reward = {reward_data['total_reward']:.1f}"
    return history