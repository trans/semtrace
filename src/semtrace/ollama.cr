require "http/client"
require "json"

module Semtrace
  # Minimal Ollama API client for getting contextual embeddings.
  module Ollama
    DEFAULT_URL = "http://localhost:11434"

    # Returns the contextual embedding for a text input from a model's forward pass.
    def self.embed(text : String, model : String = "llama3.2:3b", base_url : String = DEFAULT_URL) : Array(Float32)
      uri = URI.parse(base_url)
      client = HTTP::Client.new(uri)
      client.read_timeout = 120.seconds

      body = {model: model, input: text}.to_json

      begin
        response = client.post("/api/embed", headers: HTTP::Headers{"Content-Type" => "application/json"}, body: body)
        raise "Ollama error: HTTP #{response.status_code}" unless response.status_code == 200

        data = JSON.parse(response.body)
        embeddings = data["embeddings"][0].as_a
        embeddings.map(&.as_f.to_f32)
      ensure
        client.close
      end
    end

    # Batch embed multiple texts.
    def self.embed_batch(texts : Array(String), model : String = "llama3.2:3b", base_url : String = DEFAULT_URL) : Array(Array(Float32))
      uri = URI.parse(base_url)
      client = HTTP::Client.new(uri)
      client.read_timeout = 120.seconds

      body = {model: model, input: texts}.to_json

      begin
        response = client.post("/api/embed", headers: HTTP::Headers{"Content-Type" => "application/json"}, body: body)
        raise "Ollama error: HTTP #{response.status_code}" unless response.status_code == 200

        data = JSON.parse(response.body)
        data["embeddings"].as_a.map do |emb|
          emb.as_a.map(&.as_f.to_f32)
        end
      ensure
        client.close
      end
    end
  end
end
