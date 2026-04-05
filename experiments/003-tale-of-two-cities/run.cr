require "../../src/semtrace"

args = Semtrace::RealText.parse_args({} of String => {String, String})

text = args[:text]
label = args[:label]
data_dir = args[:data_dir].empty? ? (Path[__DIR__].parent.parent / "data").to_s : args[:data_dir]

if text.empty?
  file = Path[__DIR__].parent / "texts" / "tale-ch1.txt"
  abort "Missing #{file}" unless File.exists?(file.to_s)
  text = File.read(file.to_s)
  label = "A Tale of Two Cities, Ch. 1"
end

Semtrace::RealText.run(text, label, data_dir, trace: args[:trace])
