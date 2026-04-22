//! Prompt templates for multi-role chat formatting.
//!
//! `llama_gguf`'s `ChatEngine::chat` only accepts a single user string;
//! it has no concept of system / assistant / tool role tagging. To drive
//! `Engine::generate_streaming` with a multi-role conversation we format
//! the message list into the model's native prompt format ourselves, here
//! at the `lf-inference` layer, and hand the resulting string to the
//! engine as a plain prompt.
//!
//! Four templates are provided, one per candidate model family; see each
//! `impl PromptTemplate` for the exact wire format and how each handles
//! the `Tool` role (not all models have a native tool turn).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    /// A message representing tool output or a function-call result.
    /// How this is rendered depends on the template: models with a
    /// native tool turn use it directly; models without one fold the
    /// content into the user turn with a prefix marker.
    Tool,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// Optional name — used by some templates for tool messages to
    /// identify which tool produced the output.
    pub name: Option<String>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            name: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// A prompt template renders a sequence of messages into the single
/// string a decoder-only model expects as input.
pub trait PromptTemplate: Send + Sync {
    /// Short identifier, used for `--template` CLI selection and logs.
    fn name(&self) -> &'static str;

    /// Render `messages` into the model's prompt format.
    ///
    /// When `add_generation_prompt` is true, the returned string ends
    /// with the "assistant turn start" marker so the model begins
    /// emitting the assistant response immediately. Callers almost
    /// always want `true`; `false` is useful for unit tests of the
    /// transcript form.
    fn format(&self, messages: &[Message], add_generation_prompt: bool) -> String;
}

/// Resolve a template by its canonical name. Used by the CLI's
/// `--template` flag. Returns `None` for unknown names; `"auto"` is a
/// CLI-layer sentinel and is not resolved here.
pub fn by_name(name: &str) -> Option<Box<dyn PromptTemplate>> {
    match name {
        "chatml" => Some(Box::new(ChatMlTemplate)),
        "llama3" => Some(Box::new(Llama3Template)),
        "gemma" => Some(Box::new(GemmaTemplate)),
        "phi3" => Some(Box::new(Phi3Template)),
        _ => None,
    }
}

/// Pick a template based on a GGUF `general.architecture` string.
/// Falls back to ChatML for unknown architectures — it's the most
/// common format and the least likely to generate garbage on models
/// that were tuned on a close variant.
pub fn from_architecture(arch: &str) -> Box<dyn PromptTemplate> {
    match arch {
        "llama" => Box::new(Llama3Template),
        "gemma" | "gemma2" => Box::new(GemmaTemplate),
        "phi3" | "phi-3" | "phi2" => Box::new(Phi3Template),
        // starcoder2, qwen2, mistral, and most "chat"-tuned GGUFs use
        // ChatML or a close enough variant.
        _ => Box::new(ChatMlTemplate),
    }
}

/// ChatML format: `<|im_start|>{role}\n{content}<|im_end|>\n` per turn.
/// Used by StarCoder2, Qwen2, many instruction-tuned models. The `Tool`
/// role is rendered as an extensible turn with role `tool`; this is not
/// part of the original ChatML spec but is a well-established extension.
pub struct ChatMlTemplate;

impl PromptTemplate for ChatMlTemplate {
    fn name(&self) -> &'static str {
        "chatml"
    }

    fn format(&self, messages: &[Message], add_generation_prompt: bool) -> String {
        let mut out = String::new();
        for msg in messages {
            out.push_str("<|im_start|>");
            out.push_str(msg.role.as_str());
            if msg.role == Role::Tool {
                if let Some(name) = &msg.name {
                    out.push(' ');
                    out.push_str(name);
                }
            }
            out.push('\n');
            out.push_str(&msg.content);
            out.push_str("<|im_end|>\n");
        }
        if add_generation_prompt {
            out.push_str("<|im_start|>assistant\n");
        }
        out
    }
}

/// Llama 3 instruction format. Tool outputs use the `ipython` header
/// per Meta's convention.
pub struct Llama3Template;

impl PromptTemplate for Llama3Template {
    fn name(&self) -> &'static str {
        "llama3"
    }

    fn format(&self, messages: &[Message], add_generation_prompt: bool) -> String {
        let mut out = String::from("<|begin_of_text|>");
        for msg in messages {
            let header = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "ipython",
            };
            out.push_str("<|start_header_id|>");
            out.push_str(header);
            out.push_str("<|end_header_id|>\n\n");
            out.push_str(&msg.content);
            out.push_str("<|eot_id|>");
        }
        if add_generation_prompt {
            out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        out
    }
}

/// Gemma / CodeGemma conversation format. Gemma only knows `user` and
/// `model` turns natively, so system prompts are folded into the first
/// user turn as a prefix, and tool outputs are folded into user turns
/// with a `[Tool output]` marker.
pub struct GemmaTemplate;

impl PromptTemplate for GemmaTemplate {
    fn name(&self) -> &'static str {
        "gemma"
    }

    fn format(&self, messages: &[Message], add_generation_prompt: bool) -> String {
        let mut out = String::new();
        let mut pending_system: Option<&str> = None;

        for msg in messages {
            match msg.role {
                Role::System => {
                    pending_system = Some(&msg.content);
                }
                Role::Assistant => {
                    out.push_str("<start_of_turn>model\n");
                    out.push_str(&msg.content);
                    out.push_str("<end_of_turn>\n");
                }
                Role::User | Role::Tool => {
                    out.push_str("<start_of_turn>user\n");
                    if let Some(sys) = pending_system.take() {
                        out.push_str(sys);
                        out.push_str("\n\n");
                    }
                    if msg.role == Role::Tool {
                        out.push_str("[Tool output");
                        if let Some(name) = &msg.name {
                            out.push_str(": ");
                            out.push_str(name);
                        }
                        out.push_str("]\n");
                    }
                    out.push_str(&msg.content);
                    out.push_str("<end_of_turn>\n");
                }
            }
        }

        // If the only system message never got flushed (no user turn
        // followed it), emit it anyway as a standalone user turn so the
        // model sees it.
        if let Some(sys) = pending_system {
            out.push_str("<start_of_turn>user\n");
            out.push_str(sys);
            out.push_str("<end_of_turn>\n");
        }

        if add_generation_prompt {
            out.push_str("<start_of_turn>model\n");
        }
        out
    }
}

/// Phi-3 instruction format. Phi-3 has system / user / assistant tags
/// but no native tool turn; tool outputs are folded into user turns
/// with a `[Tool output]` marker, same as the Gemma fallback.
pub struct Phi3Template;

impl PromptTemplate for Phi3Template {
    fn name(&self) -> &'static str {
        "phi3"
    }

    fn format(&self, messages: &[Message], add_generation_prompt: bool) -> String {
        let mut out = String::new();
        for msg in messages {
            let (header, is_tool) = match msg.role {
                Role::System => ("<|system|>", false),
                Role::User => ("<|user|>", false),
                Role::Assistant => ("<|assistant|>", false),
                Role::Tool => ("<|user|>", true),
            };
            out.push_str(header);
            out.push('\n');
            if is_tool {
                out.push_str("[Tool output");
                if let Some(name) = &msg.name {
                    out.push_str(": ");
                    out.push_str(name);
                }
                out.push_str("]\n");
            }
            out.push_str(&msg.content);
            out.push_str("<|end|>\n");
        }
        if add_generation_prompt {
            out.push_str("<|assistant|>\n");
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<Message> {
        vec![
            Message::new(Role::System, "You are Ferris."),
            Message::new(Role::User, "What is ownership?"),
            Message::new(Role::Assistant, "Ownership means..."),
            Message::new(Role::Tool, "{\"rustc_version\":\"1.95.0\"}").with_name("rustc_info"),
            Message::new(Role::User, "Thanks. Now explain lifetimes."),
        ]
    }

    #[test]
    fn chatml_renders_every_role_and_adds_gen_prompt() {
        let out = ChatMlTemplate.format(&sample(), true);
        assert!(out.contains("<|im_start|>system\nYou are Ferris.<|im_end|>"));
        assert!(out.contains("<|im_start|>user\nWhat is ownership?<|im_end|>"));
        assert!(out.contains("<|im_start|>assistant\nOwnership means...<|im_end|>"));
        assert!(
            out.contains("<|im_start|>tool rustc_info\n{\"rustc_version\":\"1.95.0\"}<|im_end|>")
        );
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn llama3_uses_ipython_header_for_tool_role() {
        let out = Llama3Template.format(&sample(), true);
        assert!(out.starts_with("<|begin_of_text|>"));
        assert!(
            out.contains("<|start_header_id|>system<|end_header_id|>\n\nYou are Ferris.<|eot_id|>")
        );
        assert!(out.contains("<|start_header_id|>ipython<|end_header_id|>"));
        assert!(out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn gemma_folds_system_into_first_user_turn() {
        let out = GemmaTemplate.format(&sample(), true);
        // System should have been prepended inside the FIRST user turn,
        // not emitted standalone.
        assert!(!out.contains("<start_of_turn>system"));
        let first_user_start = out.find("<start_of_turn>user").unwrap();
        let first_user_chunk = &out[first_user_start..];
        assert!(
            first_user_chunk
                .starts_with("<start_of_turn>user\nYou are Ferris.\n\nWhat is ownership?"),
            "system prompt was not folded into the first user turn: {first_user_chunk:.200}"
        );
        assert!(out.contains("<start_of_turn>model\nOwnership means...<end_of_turn>"));
        // Tool folded into user with prefix.
        assert!(out.contains("[Tool output: rustc_info]"));
        assert!(out.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn phi3_wraps_each_turn_with_end_marker() {
        let out = Phi3Template.format(&sample(), true);
        assert!(out.contains("<|system|>\nYou are Ferris.<|end|>"));
        assert!(out.contains("<|user|>\nWhat is ownership?<|end|>"));
        assert!(out.contains("<|assistant|>\nOwnership means...<|end|>"));
        // Tool becomes a user turn with the marker line.
        assert!(out.contains(
            "<|user|>\n[Tool output: rustc_info]\n{\"rustc_version\":\"1.95.0\"}<|end|>"
        ));
        assert!(out.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn add_generation_prompt_false_omits_assistant_header() {
        let msgs = vec![Message::new(Role::User, "hi")];
        assert!(
            !ChatMlTemplate
                .format(&msgs, false)
                .contains("<|im_start|>assistant")
        );
        assert!(
            !Llama3Template
                .format(&msgs, false)
                .contains("<|start_header_id|>assistant")
        );
        assert!(
            !GemmaTemplate
                .format(&msgs, false)
                .ends_with("<start_of_turn>model\n")
        );
        assert!(
            !Phi3Template
                .format(&msgs, false)
                .ends_with("<|assistant|>\n")
        );
    }

    #[test]
    fn by_name_resolves_all_four() {
        for name in ["chatml", "llama3", "gemma", "phi3"] {
            let tmpl = by_name(name).expect(name);
            assert_eq!(tmpl.name(), name);
        }
        assert!(by_name("nonsense").is_none());
    }

    #[test]
    fn from_architecture_maps_sensibly() {
        assert_eq!(from_architecture("llama").name(), "llama3");
        assert_eq!(from_architecture("gemma").name(), "gemma");
        assert_eq!(from_architecture("gemma2").name(), "gemma");
        assert_eq!(from_architecture("phi3").name(), "phi3");
        assert_eq!(from_architecture("starcoder2").name(), "chatml");
        assert_eq!(from_architecture("qwen2").name(), "chatml");
        assert_eq!(from_architecture("unknown-arch-xyz").name(), "chatml");
    }

    #[test]
    fn gemma_flushes_dangling_system_when_no_user_follows() {
        // Safety net: if the caller builds [System] alone (weird but
        // legal), the template must still emit the system text so the
        // model sees it.
        let msgs = vec![Message::new(Role::System, "solo system")];
        let out = GemmaTemplate.format(&msgs, true);
        assert!(out.contains("<start_of_turn>user\nsolo system<end_of_turn>"));
    }
}
