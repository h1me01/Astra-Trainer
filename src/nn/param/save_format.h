#pragma once

namespace nn::param {

class SaveFormat {
  public:
    enum class Type { int8, int16, float32 };

    SaveFormat& transpose() {
        transpose_ = true;
        return *this;
    }

    SaveFormat& type(Type t) {
        type_ = t;
        return *this;
    }

    SaveFormat& scale(int s) {
        scale_ = s;
        return *this;
    }

    Type get_type() const { return type_; }
    int get_scale() const { return scale_; }
    bool is_transposed() const { return transpose_; }

  private:
    int scale_ = 1;
    bool transpose_ = false;
    Type type_ = Type::float32;
};

} // namespace nn::param
