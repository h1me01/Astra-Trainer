#pragma once

namespace nn::param {

class SaveFormat {
  public:
    enum class Type { int8, int16, float32 };

    SaveFormat& transpose() {
        m_transpose = true;
        return *this;
    }

    SaveFormat& type(Type t) {
        m_type = t;
        return *this;
    }

    SaveFormat& scale(int s) {
        m_scale = s;
        return *this;
    }

    Type get_type() const { return m_type; }
    int get_scale() const { return m_scale; }
    bool is_transposed() const { return m_transpose; }

  private:
    int m_scale = 1;
    bool m_transpose = false;
    Type m_type = Type::float32;
};

} // namespace nn::param
