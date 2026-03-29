require "./spec_helper"

describe Semtrace do
  it "has a version" do
    Semtrace::VERSION.should_not be_nil
    Semtrace::VERSION.should eq("0.1.0")
  end
end
