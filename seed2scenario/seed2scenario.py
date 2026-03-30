def attention_state(value):
    mapping = {
        0: "The person felt that they did not want to devote further attention to the event.",
        1: "The person felt that they wanted to devote a little attention to the event.",
        2: "The person felt that they wanted to devote some attention to the event.",
        3: "The person felt that they strongly wanted to devote further attention to the event."
    }
    return mapping[value]


def certainty_state(value):
    mapping = {
        0: "The person felt that they were not certain about what was happening in the situation.",
        1: "The person felt that they were only slightly certain about what was happening in the situation.",
        2: "The person felt that they were somewhat certain about what was happening in the situation.",
        3: "The person felt that they were very certain about what was happening in the situation."
    }
    return mapping[value]


def effort_state(value):
    mapping = {
        0: "The person felt that they did not need to expend mental or physical effort to deal with the situation.",
        1: "The person felt that they needed to expend a little mental or physical effort to deal with the situation.",
        2: "The person felt that they needed to expend some mental or physical effort to deal with the situation.",
        3: "The person felt that they needed to expend a great deal of mental or physical effort to deal with the situation."
    }
    return mapping[value]


def pleasantness_state(value):
    mapping = {
        0: "The person felt that the event was not pleasant.",
        1: "The person felt that the event was only slightly pleasant.",
        2: "The person felt that the event was somewhat pleasant.",
        3: "The person felt that the event was very pleasant."
    }
    return mapping[value]


def responsibility_state(value):
    mapping = {
        0: "The person felt that they were not responsible for the situation.",
        1: "The person felt that they were only slightly responsible for the situation.",
        2: "The person felt that they were somewhat responsible for the situation.",
        3: "The person felt that they were very responsible for the situation."
    }
    return mapping[value]


def self_control_state(value):
    mapping = {
        0: "The person felt that they had no ability to influence what was happening in the situation.",
        1: "The person felt that they had little ability to influence what was happening in the situation.",
        2: "The person felt that they had some ability to influence what was happening in the situation.",
        3: "The person felt that they could strongly influence what was happening in the situation."
    }
    return mapping[value]


def circumstance_state(value):
    mapping = {
        0: "The person felt that the event could have been changed or influenced by someone.",
        1: "The person felt that the event could probably have been changed or influenced by someone.",
        2: "The person felt that the event could not easily have been changed or influenced by anyone.",
        3: "The person felt that the event could not have been changed or influenced by anyone."
    }
    return mapping[value]