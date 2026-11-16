"""
Debug Pattern Detection - Find what's broken
=============================================
"""
from bot.pattern_detector import PatternDetector, PatternType

# Test case 1: Simple Fire opportunity (3 red + 1 blue, need 4th red)
test_board_1 = [
    ['R', 'R', 'R', 'B', 'G', 'G', 'Y', 'Y'],
    ['G', 'G', 'B', 'B', 'B', 'B', 'R', 'R'],
    ['Y', 'Y', 'R', 'R', 'R', 'R', 'G', 'G'],
    ['P', 'P', 'Y', 'Y', 'B', 'B', 'R', 'R'],
    ['R', 'R', 'G', 'G', 'Y', 'Y', 'P', 'P'],
    ['B', 'B', 'R', 'R', 'G', 'G', 'Y', 'Y'],
    ['P', 'P', 'Y', 'Y', 'R', 'R', 'B', 'B'],
    ['G', 'G', 'B', 'B', 'P', 'P', 'Y', 'Y']
]

print("=" * 70)
print("TEST 1: Board with known patterns")
print("=" * 70)

for i, row in enumerate(test_board_1):
    print(f"Row {i}: {' '.join(row)}")

detector = PatternDetector()
opportunities = detector.find_all_opportunities(test_board_1)

print(f"\nFound {len(opportunities)} opportunities:")
for opp in opportunities:
    print(f"  -> {opp.pattern_type.value}: {opp.description}")
    print(f"     Swap: ({opp.swap_position[0]},{opp.swap_position[1]}) -> {opp.swap_direction}")
    print(f"     Color: {opp.gem_color}")

print("\n" + "=" * 70)
print("TEST 2: Realistic board (lots of patterns)")
print("=" * 70)

# Create a board where a pattern should work
# 5-in-row void pattern
test_board_2 = [
    ['R', 'R', 'R', 'R', 'R', 'G', 'Y', 'Y'],
    ['G', 'G', 'B', 'B', 'B', 'B', 'R', 'R'],
    ['Y', 'Y', 'R', 'R', 'R', 'R', 'G', 'G'],
    ['P', 'P', 'Y', 'Y', 'B', 'B', 'R', 'R'],
    ['R', 'R', 'G', 'G', 'Y', 'Y', 'P', 'P'],
    ['B', 'B', 'R', 'R', 'G', 'G', 'Y', 'Y'],
    ['P', 'P', 'Y', 'Y', 'R', 'R', 'B', 'B'],
    ['G', 'G', 'B', 'B', 'P', 'P', 'Y', 'Y']
]

for i, row in enumerate(test_board_2):
    print(f"Row {i}: {' '.join(row)}")

opportunities_2 = detector.find_all_opportunities(test_board_2)
print(f"\nFound {len(opportunities_2)} opportunities:")
for opp in opportunities_2:
    print(f"  -> {opp.pattern_type.value}: {opp.description}")

print("\n" + "=" * 70)
print("TEST 3: Single Fire pattern (needs_swap)")
print("=" * 70)

# This should detect a Fire pattern that needs a swap
test_board_3 = [
    ['G', 'G', 'G', 'B', 'G', 'G', 'Y', 'Y'],
    ['G', 'G', 'B', 'B', 'B', 'B', 'R', 'R'],
    ['G', 'G', 'G', 'R', 'R', 'R', 'G', 'G'],
    ['P', 'P', 'Y', 'Y', 'B', 'B', 'R', 'R'],
    ['R', 'R', 'G', 'G', 'Y', 'Y', 'P', 'P'],
    ['B', 'B', 'R', 'R', 'G', 'G', 'Y', 'Y'],
    ['P', 'P', 'Y', 'Y', 'R', 'R', 'B', 'B'],
    ['G', 'G', 'B', 'B', 'P', 'P', 'Y', 'Y']
]

for i, row in enumerate(test_board_3):
    print(f"Row {i}: {' '.join(row)}")

opportunities_3 = detector.find_all_opportunities(test_board_3)
print(f"\nFound {len(opportunities_3)} opportunities:")
for opp in opportunities_3:
    print(f"  -> {opp.pattern_type.value}: {opp.description}")
    print(f"     Color: {opp.gem_color}, Position: {opp.swap_position}, Dir: {opp.swap_direction}")

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
