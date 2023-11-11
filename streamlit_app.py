# needs streamlit and python>=3.7

import torch
import urllib

import altair as alt
import dataclasses
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import streamlit as st
import numpy as np

from torch.nn.functional import softmax


np.random.seed(42)

# @st.cache
# def get_time_data():
#     pos_encoding = positional_encoding(tokens, dimensions)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding


# def softmax(xs):
#     return np.exp(xs) / sum(np.exp(xs))


def get_pos_encoding(tokens, dimensions):
    pos_encoding = positional_encoding(tokens, dimensions)
    return pos_encoding


@st.cache
def get_inputs():
    # Input 1  # Input 2  # Input 3
    # x = torch.randint(0, 3, (3, 4)).float()
    x = [[1, 0, 1, 0], [0, 2, 0, 2], [1, 1, 1, 1]]
    x = torch.tensor(x, dtype=torch.float32)
    return x


def get_wqkv(dim=None):

    w_key = [[0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0]]
    w_query = [[1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1]]
    w_value = [[0, 2, 0], [0, 3, 0], [1, 0, 3], [1, 1, 0]]
    w_key = torch.tensor(w_key, dtype=torch.float32)
    w_query = torch.tensor(w_query, dtype=torch.float32)
    w_value = torch.tensor(w_value, dtype=torch.float32)
    return w_query, w_key, w_value


def attention_scores(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """obtains the scores of the attention between Q and K

    Args:
        Q (torch.Tensor): Query matrix, shape: input_dim x attn_dim
        K (torch.Tensor): Value matrix, shape: input_dim x attn_dim

    Returns:
        torch.Tensor: attention scores, shape: attn_dim x attn_dim
    """

    # (Q * K^T).shape = attn_dim x attn_dim
    m = torch.matmul(Q, K.transpose(1, 0).float())

    # Q * K^T / sqrt(attn_dim)
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())

    # returns the softmaxed attention scores
    return m, torch.softmax(m, dim=-1)


def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Implements attention

    Args:
        Q (torch.Tensor): Query matrix, shape = input_dim x attn_dim
        K (torch.Tensor): Value matrix, shape = input_dim x attn_dim
        V (torch.Tensor): Value matrix, shape = input_dim x attn_dim

    Returns:
        torch.Tensor: [description]
    """
    # Attention(Q, K, V) = norm(QK)V
    scores, softmax_scores = attention_scores(
        Q, K
    )  # (batch_size, dim_attn, seq_length)

    return torch.matmul(softmax_scores, V)  # (batch_size, seq_length, seq_length)


@dataclasses.dataclass
class DemoState:
    x: torch.tensor
    w_q: torch.tensor
    w_k: torch.tensor
    w_v: torch.tensor


@st.cache(allow_output_mutation=True)
def persistent_demo_state() -> DemoState:

    x = get_inputs()
    w_query, w_key, w_value = get_wqkv()

    return DemoState(x, w_query, w_key, w_value)


state = persistent_demo_state()

try:
    st.write("### Attention short example")

    _rand_like = lambda input: torch.randint_like(input, 0, 3).float()
    _zeroes_like = lambda input: torch.zeros_like(input).float()

    colA, colB = st.columns(2)

    if st.button("Reset all"):
        state.x = get_inputs()
        w_query, w_key, w_value = get_wqkv()
        state.w_q = w_query
        state.w_k = w_key
        state.w_v = w_query

    with colA:
        if st.button("new x"):
            state.x = _rand_like(state.x)

        if st.button("new WQ"):
            state.w_q = _rand_like(state.w_q)

        if st.button("new WK"):
            state.w_k = _rand_like(state.w_k)

        if st.button("new WV"):
            state.w_v = _rand_like(state.w_v)

    with colB:
        if st.button("zero x"):
            state.x = _zeroes_like(state.x)

        if st.button("zero WQ"):
            state.w_q = _zeroes_like(state.w_q)

        if st.button("zero WK"):
            state.w_k = _zeroes_like(state.w_k)

        if st.button("zero WV"):
            state.w_v = _zeroes_like(state.w_v)

    queries = state.x @ state.w_q
    keys = state.x @ state.w_k
    values = state.x @ state.w_v

    # attn_scores = queries @ keys.T
    # attn_scores_softmax = softmax(attn_scores, dim=-1)

    attn_scores, attn_scores_softmax = attention_scores(Q=queries, K=keys)

    # weighted_values = values[:, None] * attn_scores_softmax.T[:, :, None]
    # outputs = weighted_values.sum(dim=0)
    outputs = attention(Q=queries, K=keys, V=values)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### input $X$")
        st.write(state.x.numpy())
    with col2:
        st.write("### $W_Q$")
        st.write(state.w_q.numpy())
        st.write("### $W_K$")
        st.write(state.w_k.numpy())
        st.write("### $W_V$")
        st.write(state.w_v.numpy())
    with col3:
        st.write("### $XW_Q$")
        st.write(queries.numpy())
        st.write("### $XW_K$")
        st.write(keys.numpy())
        st.write("### $XW_V$")
        st.write(values.numpy())

    st.write("### Attention scores")
    st.write(attn_scores.numpy())

    fig, ax = plt.subplots(figsize=(7, 4))

    im = ax.imshow(attn_scores.numpy(), cmap=plt.cm.Reds)
    ax.set_ylabel("Row")
    ax.set_xlabel("Column")

    # https://izziswift.com/matplotlib-2-subplots-1-colorbar/
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    for (i, j), z in np.ndenumerate(attn_scores.numpy()):
        ax.text(
            j, i, "{:.2f}".format(z), ha="center", va="center", color="k", fontsize=10
        )

    st.pyplot(fig)

    st.write("### Softmaxed attention scores")
    st.write(attn_scores_softmax.numpy())

    fig, ax = plt.subplots(figsize=(7, 4))

    im = ax.imshow(attn_scores_softmax.numpy(), cmap=plt.cm.Blues)
    ax.set_ylabel("Row")
    ax.set_xlabel("Column")

    for (i, j), z in np.ndenumerate(attn_scores_softmax.numpy()):
        ax.text(
            j, i, "{:.2f}".format(z), ha="center", va="center", color="k", fontsize=10
        )

    # https://izziswift.com/matplotlib-2-subplots-1-colorbar/
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    st.pyplot(fig)

    st.write("### Final output $V$")
    st.write(outputs.numpy())

    # plot
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(outputs.numpy(), cmap=plt.cm.Greens)

    # https://stackoverflow.com/questions/58332372/visualize-1d-numpy-array-as-2d-array-with-matplotlib
    for (i, j), z in np.ndenumerate(outputs.numpy()):
        ax.text(
            j, i, "{:.2f}".format(z), ha="center", va="center", color="k", fontsize=10
        )

    # https://izziswift.com/matplotlib-2-subplots-1-colorbar/
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    st.pyplot(fig)


except urllib.error.URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
    """
        % e.reason
    )

